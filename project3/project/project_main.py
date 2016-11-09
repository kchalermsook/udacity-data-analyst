import csv
import codecs
import re
import xml.etree.cElementTree as ET
import pprint

import cerberus

import schema

OSM_PATH = "some_singapore.osm"

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

SCHEMA = schema.schema

# Make sure the fields order in the csvs matches the column order in the sql table schema
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']

expected_street_key = ["Street"]
mapping_street_key = { "highawy": "highway",}

def update_name(name,mapping):
    """Update the name from mapping"""
    for key in mapping:
            if key in name:
                return name.replace(key,mapping[key] )
    return name

def update_country_name(country_name):
    """Update country name to be the same format"""
    if country_name == "Singapore":
        return "SG"
    elif country_name == "Indonesia":
        return "ID"
    elif country_name == "Malaysia":
        return  "MY"
    return country_name

def extract_house_number(house_text):
    """Convert the long house string into ony house number"""
    my_value_arr = house_text.split(" ")
    if(len(my_value_arr) > 1):
        return my_value_arr[0]
    return house_text

# Transform XML element into Python dict for node and way
def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []  

    if element.tag == 'node':
        node_attribs["id"] = element.attrib["id"]
        node_attribs["lat"] = element.attrib["lat"]
        node_attribs["lon"] = element.attrib["lon"]
        node_attribs["user"] = element.attrib["user"]
        node_attribs["uid"] = element.attrib["uid"]
        node_attribs["version"] = element.attrib["version"]
        node_attribs["changeset"] = element.attrib["changeset"]
        node_attribs["timestamp"] = element.attrib["timestamp"]
        # find child node and assign to node json
        for elem_child in element:
            elem_child_json = {}
            elem_child_json["id"] = element.attrib["id"]
            key_array = elem_child.attrib["k"].split(":")
            if(len(key_array) > 2):
                pass
            if(len(key_array) > 1):
                elem_child_json["key"] = key_array[1]
                elem_child_json["type"] =  key_array[0]
            
            else:
                elem_child_json["key"] = elem_child.attrib["k"]
                elem_child_json["type"] = "regular"
            
            elem_child_json["value"] = elem_child.attrib["v"]
            tags.append(elem_child_json)
        
        return {'node': node_attribs, 'node_tags': tags}
    elif element.tag == 'way':
        way_attribs["id"] = element.attrib["id"]
        way_attribs["user"] = element.attrib["user"]
        way_attribs["uid"] = element.attrib["uid"]
        way_attribs["version"] = element.attrib["version"]
        way_attribs["timestamp"] = element.attrib["timestamp"]
        way_attribs["changeset"] = element.attrib["changeset"]
        i = 0
        for elem_child in element:
            my_key = ""
            my_value = ""
            if elem_child.tag == "nd":
                elem_child_json = {}
                elem_child_json["id"] = element.attrib["id"]
                elem_child_json["node_id"] = elem_child.attrib["ref"]
                elem_child_json["position"] = i
                i = i + 1
                way_nodes.append(elem_child_json)
            elif elem_child.tag == "tag":
                elem_child_json = {}
                elem_child_json["id"] = element.attrib["id"]
                key_array = elem_child.attrib["k"].split(":")
                if(len(key_array) > 2):
                    pass
                    #print "xxxxxx"
                if(len(key_array) > 1):
                    my_key = elem_child.attrib["k"][len(key_array[0]) +1:]
                    my_key = update_name(my_key,mapping_street_key)
                    elem_child_json["key"] = my_key
                    elem_child_json["type"] =  key_array[0]
                else:
                    my_key = elem_child.attrib["k"]
                    elem_child_json["key"] = my_key
                    elem_child_json["type"] = "regular"
                my_value = elem_child.attrib["v"]
                
                # to change the country into same format (SG, MY, ID)
                if my_key == 'country' :
                    my_value = update_country_name(my_value)
                
                # to extract only the house number from house number with long sttring such as "45 Armenian Street"
                if my_key == 'housenumber':
                    my_value = extract_house_number(my_value)
                
                elem_child_json["value"] = my_value
                tags.append(elem_child_json)
        retVal = {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}
        return retVal


# ================================================== #
#               Helper Functions                     #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


def validate_element(element, validator, schema=SCHEMA):
    """Raise ValidationError if element does not match schema"""
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_strings = (
            "{0}: {1}".format(k, v if isinstance(v, str) else ", ".join(v))
            for k, v in errors.iteritems()
        )
        raise cerberus.ValidationError(
            message_string.format(field, "\n".join(error_strings))
        )


class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


# ================================================== #
#               Main Function                        #
# ================================================== #
def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file, \
         codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file, \
         codecs.open(WAYS_PATH, 'w') as ways_file, \
         codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file, \
         codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            #pprint.pprint(el)
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])


if __name__ == '__main__':
    # Note: Validation is ~ 10X slower. For the project consider using a small
    # sample of the map when validating.
    process_map(OSM_PATH, validate=True)