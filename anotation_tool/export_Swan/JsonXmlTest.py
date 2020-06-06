## xml to json

import json
import xmltodict

xml = """
<document>
    <annotations>
        <annotation>
            <id>50</id>
            <start>33</start>
            <end>74</end>
            <spanType>Major claim</spanType>
        </annotation>
        <annotation>
            <id>53</id>
            <start>643</start>
            <end>742</end>
            <spanType>Premise</spanType>
            <labels labelSetName="label">
                <label>proponent</label>
            </labels>
        </annotation>
        <annotation>
            <id>54</id>
            <start>744</start>
            <end>871</end>
            <spanType>Premise</spanType>
            <labels labelSetName="label">
                <label>proponent</label>
            </labels>
        </annotation>
        <annotation>
            <id>56</id>
            <start>569</start>
            <end>625</end>
            <spanType>Claim</spanType>
            <labels labelSetName="label">
                <label>proponent</label>
            </labels>
        </annotation>
        <annotation>
            <id>55</id>
            <start>3361</start>
            <end>3435</end>
            <spanType>Premise</spanType>
            <labels labelSetName="label">
                <label>opponent</label>
            </labels>
        </annotation>
        <annotation>
            <id>52</id>
            <start>485</start>
            <end>534</end>
            <spanType>Claim</spanType>
            <labels labelSetName="label">
                <label>proponent</label>
            </labels>
        </annotation>
    </annotations>
    <links>
        <link>
            <from>54</from>
            <to>56</to>
            <labels labelSetName="link">
                <label>support</label>
            </labels>
        </link>
        <link>
            <from>53</from>
            <to>56</to>
            <labels labelSetName="link">
                <label>support</label>
            </labels>
        </link>
        <link>
            <from>55</from>
            <to>56</to>
            <labels labelSetName="link">
                <label>rebuttal</label>
            </labels>
        </link>
    </links>
</document>
"""

dict = xmltodict.parse(xml)
print(dict)



result = json.dumps(dict, indent=2)
print(result)

"""
with open("test_argGraph.json", "w") as f:
    json.dump(dict, f, indent=2)
"""
