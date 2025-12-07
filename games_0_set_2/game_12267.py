import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:36:19.300977
# Source Brief: brief_02267.md
# Brief Index: 2267
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple

# Helper data structure for gears
Gear = namedtuple("Gear", ["letter", "pos", "radius", "id"])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Find words by connecting adjacent letter-gears on a grid. "
        "Submit a valid word to power up a matching machine and bring the system online."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select a path of adjacent letter-gears. "
        "Press space to submit the selected word."
    )
    auto_advance = True

    # Class attribute for difficulty progression
    DIFFICULTY_LEVEL = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # --- Visual & Style Constants ---
        self.FONT_GEAR = pygame.font.SysFont("Consolas", 22, bold=True)
        self.FONT_UI = pygame.font.SysFont("Consolas", 18, bold=True)
        self.FONT_MSG = pygame.font.SysFont("Consolas", 32, bold=True)
        
        self.COLORS = {
            "bg": (20, 25, 30),
            "bg_machinery": (30, 35, 40),
            "inactive": (70, 50, 40), # Copper
            "inactive_text": (140, 100, 80),
            "powered": (255, 215, 0), # Gold
            "powered_text": (30, 25, 20),
            "selection": (0, 191, 255), # Deep Sky Blue
            "selection_text": (240, 240, 255),
            "beam": (0, 191, 255, 150),
            "ui_text": (200, 200, 220),
            "win": (152, 251, 152),
            "loss": (255, 100, 100),
        }

        # --- Game Data ---
        self.WORD_LIST_BY_LEN = self._load_words()
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.gears = {}
        self.gear_grid = {}
        self.landscapes = []
        self.selection_path = []
        self.selection_head_id = None
        self.last_move_dir = None
        self.particles = []
        self.message_timers = []
        self.prev_space_held = False
        
        # --- Game Parameters ---
        self.MAX_STEPS = 1000
        self.GEAR_RADIUS = 20
        self.GRID_COLS = 12
        self.GRID_ROWS = 6
        self.GRID_START_X = (self.width - self.GRID_COLS * self.GEAR_RADIUS * 2.2) / 2
        self.GRID_START_Y = 120

        # Initialize state for the first time
        # self.reset() # reset() is called by the wrapper/runner
        
    def _load_words(self):
        """Hardcoded word list to avoid external file loading."""
        words = {
            "cog", "gem", "ion", "lab", "orb", "sun", "zen",
            "atom", "beam", "core", "dust", "flow", "gear", "grid", "lens",
            "node", "pipe", "port", "volt", "flux", "junk", "maze", "quark",
            "engine", "energy", "field", "forge", "fusion", "matrix", "plasma",
            "power", "prism", "relay", "steam", "switch", "system", "turbine",
            "circuit", "conduit", "crystal", "dynamo", "factory", "machine",
            "network", "reactor", "science", "voltage",
        }
        words_by_len = {}
        for word in words:
            length = len(word)
            if length not in words_by_len:
                words_by_len[length] = set()
            words_by_len[length].add(word)
        return words_by_len

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        
        self._setup_level()
        
        self.selection_path = []
        self.selection_head_id = None
        self.last_move_dir = None
        self.particles = []
        self.message_timers = []
        self.prev_space_held = False
        
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        """Generates a solvable level with landscapes and gears."""
        # 1. Create landscapes
        self.landscapes = []
        landscape_y = 40
        num_landscapes = min(GameEnv.DIFFICULTY_LEVEL, 7) # Cap difficulty
        
        possible_lengths = [l for l in self.WORD_LIST_BY_LEN.keys() if l >= 3 and l <= 7]
        chosen_lengths = self.np_random.choice(possible_lengths, num_landscapes, replace=True).tolist()
        
        total_width = num_landscapes * 100
        start_x = (self.width - total_width) / 2 + 50
        for i, length in enumerate(chosen_lengths):
            self.landscapes.append({
                "len": length, "powered": False, "pos": (start_x + i * 100, landscape_y),
                "power_anim": 0
            })

        # 2. Create gear grid and place words
        self.gears = {}
        self.gear_grid = {} # (col, row) -> gear_id
        occupied_coords = set()
        gear_id_counter = 0

        # Place letters for required words to ensure solvability
        words_to_place = [self.np_random.choice(list(self.WORD_LIST_BY_LEN[l])) for l in chosen_lengths]
        
        for word in words_to_place:
            placed = False
            for _ in range(20): # 20 attempts to place a word
                path = []
                start_col = self.np_random.integers(0, self.GRID_COLS)
                start_row = self.np_random.integers(0, self.GRID_ROWS)
                if (start_col, start_row) in occupied_coords: continue
                
                current_pos = (start_col, start_row)
                path.append(current_pos)
                
                for _ in range(len(word) - 1):
                    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    self.np_random.shuffle(moves)
                    moved = False
                    for dx, dy in moves:
                        next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                        if (0 <= next_pos[0] < self.GRID_COLS and
                            0 <= next_pos[1] < self.GRID_ROWS and
                            next_pos not in path and
                            next_pos not in occupied_coords):
                            path.append(next_pos)
                            current_pos = next_pos
                            moved = True
                            break
                    if not moved: break
                
                if len(path) == len(word):
                    for i, pos in enumerate(path):
                        occupied_coords.add(pos)
                    placed = True
                    break
            
            # If successfully found a path, place the letters
            if placed:
                for i, (col, row) in enumerate(path):
                    x = self.GRID_START_X + col * self.GEAR_RADIUS * 2.2
                    y = self.GRID_START_Y + row * self.GEAR_RADIUS * 2.2
                    gear = Gear(word[i], (x, y), self.GEAR_RADIUS, gear_id_counter)
                    self.gears[gear_id_counter] = gear
                    self.gear_grid[(col, row)] = gear_id_counter
                    gear_id_counter += 1

        # 3. Fill remaining grid spots with random letters
        vowels = "aeiou"
        consonants = "bcdfghjklmnpqrstvwxyz"
        num_to_fill = (self.GRID_COLS * self.GRID_ROWS) // 2 - len(self.gears)
        
        for _ in range(num_to_fill):
            for attempt in range(10):
                col, row = self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS)
                if (col, row) not in self.gear_grid:
                    letter = self.np_random.choice(list(vowels if self.np_random.random() < 0.4 else consonants))
                    x = self.GRID_START_X + col * self.GEAR_RADIUS * 2.2
                    y = self.GRID_START_Y + row * self.GEAR_RADIUS * 2.2
                    gear = Gear(letter, (x, y), self.GEAR_RADIUS, gear_id_counter)
                    self.gears[gear_id_counter] = gear
                    self.gear_grid[(col, row)] = gear_id_counter
                    gear_id_counter += 1
                    break
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        space_held = action[1] == 1
        # shift_held = action[2] == 1 # For future power-ups
        
        reward = 0.0
        terminated = False
        
        # --- Handle Input ---
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        if movement != 0:
            reward += self._handle_movement(movement)
        
        if space_pressed:
            reward += self._handle_teleport()

        # --- Update Game Logic ---
        self.steps += 1
        self._update_particles()
        self._update_animations()
        
        # --- Check Termination Conditions ---
        if all(l["powered"] for l in self.landscapes):
            # --- Sound: Win ---
            self.game_over = True
            terminated = True
            reward += 100
            self.win_message = "SYSTEM POWERED"
            GameEnv.DIFFICULTY_LEVEL += 1
        elif len(self.gears) == 0 or self.steps >= self.MAX_STEPS:
            # --- Sound: Loss ---
            self.game_over = True
            terminated = True
            reward -= 100
            self.win_message = "INSUFFICIENT PARTS" if len(self.gears) == 0 else "SYSTEM TIMEOUT"

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement_action):
        """Updates the selection path based on movement."""
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # up, down, left, right
        reverse_map = {1: 2, 2: 1, 3: 4, 4: 3}
        
        # Backtracking
        if self.selection_path and movement_action == reverse_map.get(self.last_move_dir):
            # --- Sound: Selection backtrack ---
            self.selection_path.pop()
            if self.selection_path:
                self.selection_head_id = self.selection_path[-1]
            else:
                self.selection_head_id = None
            self.last_move_dir = None # Reset direction after backtracking
            return -0.05 # Small penalty for backtracking

        if not self.selection_path:
            # Start a new path: find the first available gear in the chosen direction from center
            start_pos = (self.GRID_COLS // 2, self.GRID_ROWS // 2)
            for i in range(1, max(self.GRID_COLS, self.GRID_ROWS)):
                check_pos = (start_pos[0] + move_map[movement_action][0] * i,
                             start_pos[1] + move_map[movement_action][1] * i)
                if check_pos in self.gear_grid:
                    gear_id = self.gear_grid[check_pos]
                    self.selection_path.append(gear_id)
                    self.selection_head_id = gear_id
                    self.last_move_dir = movement_action
                    # --- Sound: Selection start ---
                    return 0.1
            return -0.1 # No gear found in that direction

        # Extend the path
        head_gear_pos = self._get_gear_coord(self.selection_head_id)
        if head_gear_pos is None: return -0.1

        next_coord = (head_gear_pos[0] + move_map[movement_action][0],
                      head_gear_pos[1] + move_map[movement_action][1])
        
        if next_coord in self.gear_grid:
            next_gear_id = self.gear_grid[next_coord]
            if next_gear_id not in self.selection_path:
                self.selection_path.append(next_gear_id)
                self.selection_head_id = next_gear_id
                self.last_move_dir = movement_action
                # --- Sound: Selection extend ---
                return 0.1
        
        return -0.1 # Invalid move

    def _handle_teleport(self):
        """Processes the word submission."""
        if not self.selection_path:
            return 0
        
        word = "".join([self.gears[gid].letter for gid in self.selection_path])
        word_len = len(word)
        reward = 0
        
        # --- Sound: Teleport attempt ---
        
        is_valid_word = word in self.WORD_LIST_BY_LEN.get(word_len, set())
        
        if not is_valid_word:
            reward = -1.0
            self._add_message(f"'{word.upper()}' UNKNOWN", self.COLORS["loss"])
        else:
            # Find a matching, unpowered landscape
            target_landscape = None
            for l in self.landscapes:
                if l["len"] == word_len and not l["powered"]:
                    target_landscape = l
                    break
            
            if target_landscape:
                # SUCCESS!
                reward = 5.0
                target_landscape["powered"] = True
                target_landscape["power_anim"] = 30 # 1 second animation
                # --- Sound: Power up success ---
                self._add_message(f"'{word.upper()}' -> POWERED", self.COLORS["win"])
                
                # Remove used gears and create particles
                path_gears = [self.gears[gid] for gid in self.selection_path]
                for gear in path_gears:
                    self._create_teleport_particles(gear.pos, target_landscape["pos"])
                    del self.gears[gear.id]
                    coord = self._get_gear_coord(gear.id)
                    if coord:
                        del self.gear_grid[coord]
            else:
                # Valid word, but no matching landscape
                reward = -2.0
                self._add_message(f"NO {word_len}-LETTER SLOT", self.COLORS["loss"])
                # --- Sound: Misfire/error ---
        
        # Reset selection after every attempt
        self.selection_path = []
        self.selection_head_id = None
        self.last_move_dir = None
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLORS["bg"])
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Rendering Methods ---
    
    def _render_game(self):
        self._draw_background_machinery()
        self._draw_landscapes_and_connections()
        self._draw_gears()
        self._draw_selection_beam()
        self._draw_particles()

    def _render_ui(self):
        # Score and Steps
        score_text = self.FONT_UI.render(f"SCORE: {self.score:.1f}", True, self.COLORS["ui_text"])
        steps_text = self.FONT_UI.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLORS["ui_text"])
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (self.width - steps_text.get_width() - 10, 10))
        
        # Current word
        if self.selection_path:
            word = "".join([self.gears[gid].letter for gid in self.selection_path])
            word_text = self.FONT_GEAR.render(word.upper(), True, self.COLORS["selection"])
            self.screen.blit(word_text, (self.width // 2 - word_text.get_width() // 2, self.height - 30))

        # Messages
        for i, (text, color, timer) in reversed(list(enumerate(self.message_timers))):
            alpha = min(255, int(255 * (timer / 60.0)))
            msg_surf = self.FONT_UI.render(text, True, color)
            msg_surf.set_alpha(alpha)
            self.screen.blit(msg_surf, (self.width // 2 - msg_surf.get_width() // 2, 85 + i*20))
        
        # Game Over Message
        if self.game_over:
            color = self.COLORS["win"] if "POWERED" in self.win_message else self.COLORS["loss"]
            msg_surf = self.FONT_MSG.render(self.win_message, True, color)
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            self.screen.blit(msg_surf, (self.width//2 - msg_surf.get_width()//2, self.height//2 - msg_surf.get_height()//2))

    def _draw_background_machinery(self):
        for i in range(0, self.width, 40):
            pygame.draw.line(self.screen, self.COLORS["bg_machinery"], (i, 0), (i, self.height), 1)
        for i in range(0, self.height, 40):
            pygame.draw.line(self.screen, self.COLORS["bg_machinery"], (0, i), (self.width, i), 1)

    def _draw_landscapes_and_connections(self):
        central_machine_pos = (self.width / 2, 85)
        
        for l in self.landscapes:
            pos = l["pos"]
            color = self.COLORS["powered"] if l["powered"] else self.COLORS["inactive"]
            text_color = self.COLORS["powered_text"] if l["powered"] else self.COLORS["inactive_text"]
            
            # Connection line
            line_color = self.COLORS["powered"] if l["powered"] else self.COLORS["bg_machinery"]
            pygame.gfxdraw.line(self.screen, int(pos[0]), int(pos[1]), int(central_machine_pos[0]), int(central_machine_pos[1]), line_color)
            
            # Animated power flow
            if l["powered"]:
                anim_progress = ((self.steps * 2) % 100) / 100.0
                p_x = pos[0] + (central_machine_pos[0] - pos[0]) * anim_progress
                p_y = pos[1] + (central_machine_pos[1] - pos[1]) * anim_progress
                pygame.gfxdraw.filled_circle(self.screen, int(p_x), int(p_y), 3, self.COLORS["powered"])
            
            # Landscape box
            rect = pygame.Rect(pos[0] - 30, pos[1] - 15, 60, 30)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            
            # Power-up animation glow
            if l["power_anim"] > 0:
                glow_radius = 30 + (30 - l["power_anim"])
                glow_alpha = int(100 * (l["power_anim"] / 30.0))
                s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.COLORS["powered"], glow_alpha), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(s, (pos[0]-glow_radius, pos[1]-glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

            # Text
            text = self.FONT_UI.render(str(l["len"]), True, text_color)
            self.screen.blit(text, (pos[0] - text.get_width() // 2, pos[1] - text.get_height() // 2))

        # Central machine
        powered_count = sum(1 for l in self.landscapes if l["powered"])
        total_count = len(self.landscapes)
        base_color = self.COLORS["powered"] if powered_count == total_count else self.COLORS["inactive"]
        pygame.gfxdraw.filled_circle(self.screen, int(central_machine_pos[0]), int(central_machine_pos[1]), 15, base_color)
        pygame.gfxdraw.aacircle(self.screen, int(central_machine_pos[0]), int(central_machine_pos[1]), 15, base_color)
        
    def _draw_gears(self):
        for gear in self.gears.values():
            is_selected = gear.id in self.selection_path
            color = self.COLORS["selection"] if is_selected else self.COLORS["inactive"]
            text_color = self.COLORS["selection_text"] if is_selected else self.COLORS["inactive_text"]
            self._draw_gear_shape(gear.pos, gear.radius, 8, color)
            
            letter_surf = self.FONT_GEAR.render(gear.letter.upper(), True, text_color)
            self.screen.blit(letter_surf, (gear.pos[0] - letter_surf.get_width() / 2, gear.pos[1] - letter_surf.get_height() / 2))

    def _draw_gear_shape(self, center, radius, num_teeth, color):
        # Draw the main body of the gear
        pygame.gfxdraw.filled_circle(self.screen, int(center[0]), int(center[1]), radius, color)
        pygame.gfxdraw.aacircle(self.screen, int(center[0]), int(center[1]), radius, color)
        # Draw the inner hole
        pygame.gfxdraw.filled_circle(self.screen, int(center[0]), int(center[1]), int(radius*0.4), self.COLORS["bg"])

        # Draw teeth
        tooth_radius = radius + 4
        for i in range(num_teeth):
            angle = (i / num_teeth) * 2 * math.pi
            p1_angle = angle - 0.1 / num_teeth
            p2_angle = angle + 0.1 / num_teeth
            
            p1 = (center[0] + radius * math.cos(p1_angle), center[1] + radius * math.sin(p1_angle))
            p2 = (center[0] + tooth_radius * math.cos(p1_angle), center[1] + tooth_radius * math.sin(p1_angle))
            p3 = (center[0] + tooth_radius * math.cos(p2_angle), center[1] + tooth_radius * math.sin(p2_angle))
            p4 = (center[0] + radius * math.cos(p2_angle), center[1] + radius * math.sin(p2_angle))
            
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], color)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], color)

    def _draw_selection_beam(self):
        if len(self.selection_path) > 1:
            points = [self.gears[gid].pos for gid in self.selection_path]
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i+1]
                
                # Draw a "glowing" line
                for j in range(5, 0, -1):
                    alpha = 150 - j * 25
                    pygame.gfxdraw.line(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), (*self.COLORS["selection"], alpha))

    def _draw_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            
            radius = int(p["radius"] * (p["life"] / p["max_life"]))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), radius, p["color"])

    # --- Helper & Update Methods ---

    def _get_gear_coord(self, gear_id):
        for coord, gid in self.gear_grid.items():
            if gid == gear_id:
                return coord
        return None

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _update_animations(self):
        for l in self.landscapes:
            if l["power_anim"] > 0:
                l["power_anim"] -= 1
        self.message_timers = [(t, c, time-1) for t, c, time in self.message_timers if time > 0]

    def _add_message(self, text, color):
        self.message_timers.append([text, color, 90]) # 3 second message
        if len(self.message_timers) > 3:
            self.message_timers.pop(0)

    def _create_teleport_particles(self, start_pos, end_pos):
        for _ in range(30):
            life = self.np_random.integers(20, 40)
            self.particles.append({
                "pos": list(start_pos),
                "vel": [
                    (end_pos[0] - start_pos[0]) / life + self.np_random.uniform(-1, 1),
                    (end_pos[1] - start_pos[1]) / life + self.np_random.uniform(-1, 1)
                ],
                "life": life,
                "max_life": life,
                "radius": self.np_random.integers(2, 5),
                "color": self.COLORS["powered"]
            })

    def render(self):
        return self._get_observation()

# Example of how to run the environment
if __name__ == '__main__':
    # The `validate_implementation` was removed as it's not part of the standard API
    # and can cause issues if not maintained. The provided tests are sufficient.
    
    # Set a non-dummy driver for local testing
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Controls ---
    # Arrow Keys: Move selection
    # Space: Teleport word
    # Q: Quit
    
    running = True
    terminated = False
    
    # Use a separate display for rendering if not just getting rgb_array
    display_screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Cog-Words")
    
    # Action array: [movement, space, shift]
    action = [0, 0, 0]

    while running:
        # Reset movement action each frame
        action[0] = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                # Map keys to actions
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
        
        # Handle held keys
        keys = pygame.key.get_pressed()
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        if terminated:
            # Wait for a key press to reset after game over
            if any(keys):
                obs, info = env.reset()
                terminated = False
        else:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for smooth play

    pygame.quit()