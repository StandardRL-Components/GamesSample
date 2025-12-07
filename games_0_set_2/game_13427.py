import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T21:33:09.493224
# Source Brief: brief_03427.md
# Brief Index: 3427
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import string

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent learns to throw a javelin.
    The goal is to maximize the total distance thrown by hitting letters
    of a target word in sequence.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Throw a javelin to hit letters of a target word in sequence, unlocking new javelins as you "
        "increase your total distance."
    )
    user_guide = (
        "Controls: Use ↑↓←→ arrow keys to aim angle and power. Press space to throw the javelin. "
        "Press shift to cycle javelin types."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000

    # --- Colors ---
    COLOR_BG_SKY = (10, 20, 40)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_ACCENT = (100, 150, 255)
    COLOR_UI_POWER_BG = (50, 50, 80)
    COLOR_UI_POWER_FILL = (255, 200, 50)
    COLOR_TARGET = (150, 160, 180)
    COLOR_TARGET_TEXT = (20, 30, 50)
    COLOR_TARGET_HIT_CORRECT = (100, 255, 100)
    COLOR_TARGET_HIT_WRONG = (255, 100, 100)
    COLOR_TARGET_NEXT = (255, 255, 100)

    # --- Game Physics & Parameters ---
    LAUNCHER_POS = (60, SCREEN_HEIGHT - 80)
    MIN_ANGLE, MAX_ANGLE = -60, 60
    MIN_POWER, MAX_POWER = 20, 100
    JPM = 5  # Pixels per meter for distance calculation

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 48)

        # --- Persistent State (across episodes) ---
        self.persistent_total_distance = 0
        self.unlocked_javelins = {"standard": {"color": (255, 80, 80), "gravity_mod": 1.0}}
        self.initial_javelin_inventory = {"standard": 10}

        # --- Initialize state variables (will be properly set in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_state = "AIMING"  # AIMING or FLIGHT
        self.throw_angle = 0
        self.throw_power = (self.MIN_POWER + self.MAX_POWER) / 2
        self.javelin_inventory = {}
        self.javelin_types = []
        self.current_javelin_type_idx = 0
        self.active_javelins = []
        self.terrain_points = []
        self.planet_gravity = 0
        self.planet_color_ground = (0, 0, 0)
        self.targets = []
        self.target_word = ""
        self.target_word_progress = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Episode State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_state = "AIMING"
        self.active_javelins = []
        self.throw_angle = 0
        self.throw_power = (self.MIN_POWER + self.MAX_POWER) / 2
        self.prev_space_held = False
        self.prev_shift_held = False

        # --- Progression-based Setup ---
        self._update_progression()
        self.javelin_inventory = self.initial_javelin_inventory.copy()
        self.javelin_types = list(self.unlocked_javelins.keys())
        self.current_javelin_type_idx = 0

        # --- Procedural Generation ---
        self._generate_planet()
        self._generate_targets()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        if self.game_state == "AIMING":
            self._handle_aiming_input(movement, space_held, shift_held)
        
        # Javelin cycling can happen anytime
        if shift_held and not self.prev_shift_held:
            self._cycle_javelin_type()
            # SFX: UI_Switch.wav
        
        # Check for javelin throw action
        if space_held and not self.prev_space_held and self.game_state == "AIMING":
            reward += self._throw_javelin()
            # SFX: Javelin_Throw.wav

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        if self.game_state == "FLIGHT":
            reward += self._update_flight()

        self.steps += 1
        terminated = self.steps >= self.MAX_STEPS

        # Clamp reward as per spec
        final_reward = np.clip(reward, -10.0, 10.0)
        self.score += final_reward

        return self._get_observation(), final_reward, terminated, False, self._get_info()

    # --- Core Logic Sub-routines ---

    def _update_progression(self):
        """Update game difficulty and unlocks based on persistent total distance."""
        # Unlock 'Heavy' javelin
        if self.persistent_total_distance > 2000 and "heavy" not in self.unlocked_javelins:
            self.unlocked_javelins["heavy"] = {"color": (100, 100, 255), "gravity_mod": 1.5}
            self.initial_javelin_inventory["heavy"] = 5
        # Unlock 'Light' javelin
        if self.persistent_total_distance > 8000 and "light" not in self.unlocked_javelins:
            self.unlocked_javelins["light"] = {"color": (255, 255, 100), "gravity_mod": 0.6}
            self.initial_javelin_inventory["light"] = 5

    def _generate_planet(self):
        """Generate terrain, gravity, and colors for the current planet."""
        # Difficulty scaling
        self.planet_gravity = 9.8 + 0.5 * (self.persistent_total_distance // 5000)
        
        # Use total distance to seed planet appearance for variety
        planet_seed = int(self.persistent_total_distance / 1000)
        r = random.Random(planet_seed)
        
        self.planet_color_ground = (
            r.randint(20, 60), r.randint(30, 70), r.randint(40, 80)
        )
        
        self.terrain_points = []
        y = self.SCREEN_HEIGHT - 60
        for x in range(0, self.SCREEN_WIDTH + 10, 10):
            y += r.uniform(-5, 5)
            y = np.clip(y, self.SCREEN_HEIGHT - 120, self.SCREEN_HEIGHT - 20)
            self.terrain_points.append((x, int(y)))
        self.terrain_points.append((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.terrain_points.append((0, self.SCREEN_HEIGHT))

    def _generate_targets(self):
        """Generate a target word and place letters in the world."""
        word_length = 3 + (self.persistent_total_distance // 2000)
        word_length = min(word_length, 8)
        self.target_word = "".join(self.np_random.choice(list(string.ascii_uppercase), word_length))
        self.target_word_progress = 0
        self.targets = []

        available_x = list(range(int(self.LAUNCHER_POS[0] + 150), self.SCREEN_WIDTH - 50, 80))
        random.shuffle(available_x)

        for i, letter in enumerate(self.target_word):
            if not available_x: break
            x = available_x.pop(0)
            
            # Find terrain height at x
            y = self.SCREEN_HEIGHT
            for j in range(len(self.terrain_points) - 2):
                if self.terrain_points[j][0] <= x < self.terrain_points[j+1][0]:
                    y = self.terrain_points[j][1]
                    break
            
            target_y = y - 40 # Place target above terrain
            
            self.targets.append({
                "letter": letter,
                "pos": (x, target_y),
                "rect": pygame.Rect(x - 15, target_y - 15, 30, 30),
                "hit": False,
                "hit_correct": False
            })

    def _handle_aiming_input(self, movement, space_held, shift_held):
        """Adjust angle and power based on player input."""
        power_step = 1.0
        angle_step = 1.5

        if movement == 1: # Up
            self.throw_power += power_step
        elif movement == 2: # Down
            self.throw_power -= power_step
        elif movement == 3: # Left
            self.throw_angle -= angle_step
        elif movement == 4: # Right
            self.throw_angle += angle_step

        self.throw_power = np.clip(self.throw_power, self.MIN_POWER, self.MAX_POWER)
        self.throw_angle = np.clip(self.throw_angle, self.MIN_ANGLE, self.MAX_ANGLE)

    def _cycle_javelin_type(self):
        """Cycle through available javelin types."""
        if len(self.javelin_types) > 1:
            self.current_javelin_type_idx = (self.current_javelin_type_idx + 1) % len(self.javelin_types)

    def _throw_javelin(self):
        """Create a javelin entity and transition to FLIGHT state."""
        javelin_type_name = self.javelin_types[self.current_javelin_type_idx]
        
        if self.javelin_inventory.get(javelin_type_name, 0) > 0:
            self.javelin_inventory[javelin_type_name] -= 1
            
            angle_rad = math.radians(self.throw_angle)
            vel_x = self.throw_power * math.cos(angle_rad)
            vel_y = -self.throw_power * math.sin(angle_rad) # Negative because pygame y is inverted
            
            javelin_spec = self.unlocked_javelins[javelin_type_name]
            
            self.active_javelins.append({
                "pos": list(self.LAUNCHER_POS),
                "vel": [vel_x, vel_y],
                "type": javelin_type_name,
                "spec": javelin_spec,
                "trail": [],
                "distance": 0,
                "last_pos": list(self.LAUNCHER_POS)
            })
            self.game_state = "FLIGHT"
            return 0 # No immediate reward for throwing
        else:
            # Out of all javelins, reset inventory with penalty
            self.javelin_inventory = self.initial_javelin_inventory.copy()
            # SFX: Error.wav
            return -1.0

    def _update_flight(self):
        """Update physics, collisions, and state for flying javelins."""
        if not self.active_javelins:
            self.game_state = "AIMING"
            return 0

        total_reward = 0
        
        for javelin in self.active_javelins[:]:
            # --- Physics Update ---
            javelin["last_pos"] = list(javelin["pos"])
            gravity_effect = self.planet_gravity * javelin["spec"]["gravity_mod"]
            javelin["vel"][1] += gravity_effect * (1 / self.FPS)
            javelin["pos"][0] += javelin["vel"][0] * (1 / self.FPS)
            javelin["pos"][1] += javelin["vel"][1] * (1 / self.FPS)

            step_dist = math.hypot(javelin["pos"][0] - javelin["last_pos"][0], javelin["pos"][1] - javelin["last_pos"][1])
            javelin["distance"] += step_dist / self.JPM
            total_reward += 0.1 * (step_dist / self.JPM)

            # --- Trail Update ---
            javelin["trail"].append({
                "pos": list(javelin["pos"]), "life": 1.0, "size": 6
            })
            for particle in javelin["trail"]:
                particle["life"] -= 0.02
                particle["size"] = max(0, particle["life"] * 6)
            javelin["trail"] = [p for p in javelin["trail"] if p["life"] > 0]

            # --- Collision Detection ---
            collided = False
            # Terrain collision
            if javelin["pos"][0] >= 0 and javelin["pos"][0] < self.SCREEN_WIDTH:
                terrain_y = self.SCREEN_HEIGHT
                for i in range(len(self.terrain_points) - 2):
                    p1 = self.terrain_points[i]
                    p2 = self.terrain_points[i+1]
                    if p1[0] <= javelin["pos"][0] < p2[0]:
                        # Linear interpolation for terrain height
                        ratio = (javelin["pos"][0] - p1[0]) / (p2[0] - p1[0])
                        terrain_y = p1[1] + ratio * (p2[1] - p1[1])
                        break
                if javelin["pos"][1] > terrain_y:
                    collided = True
                    # SFX: Javelin_Hit_Ground.wav

            # Out of bounds
            if not (0 < javelin["pos"][0] < self.SCREEN_WIDTH):
                collided = True

            # Target collision
            javelin_rect = pygame.Rect(javelin["pos"][0]-2, javelin["pos"][1]-2, 4, 4)
            for target in self.targets:
                if not target["hit"] and target["rect"].colliderect(javelin_rect):
                    target["hit"] = True
                    # SFX: Target_Hit.wav
                    total_reward += 1.0
                    
                    if self.target_word_progress < len(self.target_word) and \
                       target["letter"] == self.target_word[self.target_word_progress]:
                        target["hit_correct"] = True
                        self.target_word_progress += 1
                        total_reward += 5.0
                        # SFX: Target_Hit_Correct.wav
                        
                        # Word complete bonus
                        if self.target_word_progress == len(self.target_word):
                            total_reward += 10.0
                            # SFX: Word_Complete.wav
                    
                    # Javelin passes through targets
            
            if collided:
                self.persistent_total_distance += javelin["distance"]
                self.active_javelins.remove(javelin)

        if not self.active_javelins:
            self.game_state = "AIMING"
            if self.target_word_progress == len(self.target_word):
                # Reset targets for next throw if word was completed
                self._generate_targets()

        return total_reward

    # --- Rendering Sub-routines ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG_SKY)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render terrain
        pygame.draw.polygon(self.screen, self.planet_color_ground, self.terrain_points)

        # Render targets
        for target in self.targets:
            color = self.COLOR_TARGET
            if target["hit"]:
                color = self.COLOR_TARGET_HIT_CORRECT if target["hit_correct"] else self.COLOR_TARGET_HIT_WRONG
            elif self.target_word_progress < len(self.target_word) and target["letter"] == self.target_word[self.target_word_progress]:
                # Glow effect for the next target
                glow_size = 18 + 4 * math.sin(self.steps * 0.2)
                pygame.gfxdraw.filled_circle(self.screen, int(target["pos"][0]), int(target["pos"][1]), int(glow_size), (*self.COLOR_TARGET_NEXT, 80))

            pygame.gfxdraw.filled_circle(self.screen, int(target["pos"][0]), int(target["pos"][1]), 15, color)
            pygame.gfxdraw.aacircle(self.screen, int(target["pos"][0]), int(target["pos"][1]), 15, color)
            
            letter_surf = self.font_medium.render(target["letter"], True, self.COLOR_TARGET_TEXT)
            self.screen.blit(letter_surf, letter_surf.get_rect(center=target["pos"]))

        # Render javelin trails and javelins
        for javelin in self.active_javelins:
            # Trail
            for particle in javelin["trail"]:
                alpha = int(particle["life"] * 200)
                color = (*javelin["spec"]["color"], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(particle["pos"][0]), int(particle["pos"][1]), int(particle["size"]), color)
            
            # Javelin
            angle_rad = math.atan2(-javelin["vel"][1], javelin["vel"][0])
            j_len = 20
            end_x = javelin["pos"][0] + j_len * math.cos(angle_rad)
            end_y = javelin["pos"][1] + j_len * math.sin(angle_rad)
            pygame.draw.line(self.screen, javelin["spec"]["color"], javelin["pos"], (end_x, end_y), 3)

    def _render_ui(self):
        # --- Aiming Reticle (if aiming) ---
        if self.game_state == "AIMING":
            # Power bar
            bar_w, bar_h = 150, 20
            bar_x, bar_y = self.LAUNCHER_POS[0] - bar_w / 2, self.LAUNCHER_POS[1] + 20
            power_ratio = (self.throw_power - self.MIN_POWER) / (self.MAX_POWER - self.MIN_POWER)
            pygame.draw.rect(self.screen, self.COLOR_UI_POWER_BG, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_UI_POWER_FILL, (bar_x, bar_y, bar_w * power_ratio, bar_h))
            
            # Angle indicator
            angle_rad = math.radians(self.throw_angle)
            line_len = 50
            end_x = self.LAUNCHER_POS[0] + line_len * math.cos(angle_rad)
            end_y = self.LAUNCHER_POS[1] - line_len * math.sin(angle_rad)
            pygame.draw.line(self.screen, self.COLOR_UI_ACCENT, self.LAUNCHER_POS, (end_x, end_y), 2)
            pygame.gfxdraw.filled_circle(self.screen, int(self.LAUNCHER_POS[0]), int(self.LAUNCHER_POS[1]), 8, self.COLOR_UI_ACCENT)

        # --- Top Left: Score/Distance ---
        dist_text = f"TOTAL DISTANCE: {int(self.persistent_total_distance)}m"
        dist_surf = self.font_medium.render(dist_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(dist_surf, (10, 10))

        # --- Top Right: Target Word ---
        x_offset = self.SCREEN_WIDTH - 10
        for i, letter in reversed(list(enumerate(self.target_word))):
            color = self.COLOR_UI_TEXT
            if i < self.target_word_progress:
                color = self.COLOR_TARGET_HIT_CORRECT
            elif i == self.target_word_progress:
                color = self.COLOR_TARGET_NEXT
            
            letter_surf = self.font_large.render(letter, True, color)
            x_offset -= letter_surf.get_width() + 5
            self.screen.blit(letter_surf, (x_offset, 10))

        # --- Bottom Left: Javelin Info ---
        javelin_type_name = self.javelin_types[self.current_javelin_type_idx]
        javelin_spec = self.unlocked_javelins[javelin_type_name]
        count = self.javelin_inventory.get(javelin_type_name, 0)
        
        type_text = f"TYPE: {javelin_type_name.upper()}"
        count_text = f"REMAINING: {count}"
        
        type_surf = self.font_small.render(type_text, True, javelin_spec["color"])
        count_surf = self.font_small.render(count_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(type_surf, (10, self.SCREEN_HEIGHT - 45))
        self.screen.blit(count_surf, (10, self.SCREEN_HEIGHT - 25))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "total_distance": self.persistent_total_distance,
            "target_word_progress": f"{self.target_word_progress}/{len(self.target_word)}"
        }

    def close(self):
        pygame.quit()

# Example usage:
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # Use this to manually play the game
    import pygame
    
    # Re-initialize pygame for display
    pygame.init()
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Javelin Zero")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    # --- Control mapping for human play ---
    # ARROWS: Move angle/power
    # SPACE: Throw
    # SHIFT: Cycle javelin type
    
    while not done:
        # Default action is "do nothing"
        movement = 0
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Total Dist: {info['total_distance']:.2f}")

        if terminated or truncated:
            print("Episode finished!")
            obs, info = env.reset()

        # --- Display the rendered frame ---
        # The observation is (H, W, C), but pygame surfaces are (W, H)
        # So we need to transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()