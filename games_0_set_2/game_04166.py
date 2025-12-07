
# Generated: 2025-08-28T01:37:03.233741
# Source Brief: brief_04166.md
# Brief Index: 4166

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place an item. Shift to cycle between items."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist Zen garden puzzle. Place rocks and plants to achieve a target aesthetic score before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.FPS = 30
        
        # Visuals & Game Constants
        self.COLOR_BG = (240, 235, 220)
        self.COLOR_GRID = (220, 215, 200)
        self.COLOR_CURSOR = (255, 255, 150, 100)
        self.COLOR_ROCK_FACE = (100, 100, 110)
        self.COLOR_ROCK_TOP = (120, 120, 130)
        self.COLOR_PLANT_STEM = (100, 140, 90)
        self.COLOR_PLANT_LEAF = (120, 180, 110)
        self.COLOR_TEXT = (80, 80, 80)
        self.COLOR_MSG_BG = (255, 255, 255, 200)

        self.GRID_W, self.GRID_H = 12, 8
        self.TILE_W, self.TILE_H = 48, 24
        self.ORIGIN_X, self.ORIGIN_Y = self.WIDTH // 2, 120
        
        self.FONT_UI = pygame.font.SysFont("sans-serif", 24)
        self.FONT_MSG = pygame.font.SysFont("sans-serif", 48, bold=True)
        
        self.available_elements_defs = {
            "rock_s": {"type": "rock", "size": 0.6},
            "rock_m": {"type": "rock", "size": 0.8},
            "plant_s": {"type": "plant", "size": 0.7, "sway": 2},
            "plant_l": {"type": "plant", "size": 1.0, "sway": 3},
        }
        self.available_elements_keys = list(self.available_elements_defs.keys())
        
        # State variables initialized in reset
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.level = None
        self.time_remaining = None
        self.target_score = None
        self.placed_elements = None
        self.particles = None
        self.cursor_pos = None
        self.selected_element_index = None
        self.last_space_held = None
        self.last_shift_held = None
        self.message = None
        self.message_timer = None
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        self.steps = 0
        self.game_over = False
        self.win = False
        self.level = 1
        self.message = ""
        self.message_timer = 0
        
        self._setup_level()
        
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def _setup_level(self):
        self.time_remaining = 60 * self.FPS
        self.score = 0
        if self.level == 1: self.target_score = 50
        elif self.level == 2: self.target_score = 75
        else: self.target_score = 100
        
        self.placed_elements = []
        self.particles = []
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_element_index = 0

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.02  # Time penalty

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # Handle input
        reward += self._handle_input(movement, space_held, shift_held)
        
        # Update game state
        self._update_game_state()

        terminated = False
        if self.score >= self.target_score and not self.message:
            reward += 100
            self.message_timer = self.FPS * 2
            if self.level < 3:
                self.message = f"Level {self.level} Complete!"
            else:
                self.message = "Zen Master!"
                self.win = True
                self.game_over = True
                terminated = True
        
        if self.time_remaining <= 0 and not self.game_over:
            reward -= 100
            self.game_over = True
            terminated = True
            self.message = "Time's Up!"
            self.message_timer = self.FPS * 3

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        
        # --- Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        # --- Placement (on key press) ---
        if space_held and not self.last_space_held:
            reward += self._place_element()
        self.last_space_held = space_held
        
        # --- Cycle Element (on key press) ---
        if shift_held and not self.last_shift_held:
            self.selected_element_index = (self.selected_element_index + 1) % len(self.available_elements_keys)
        self.last_shift_held = shift_held
        
        return reward

    def _update_game_state(self):
        # Countdown timer and level transitions
        if not self.message:
            self.time_remaining = max(0, self.time_remaining - 1)
        
        if self.message_timer > 0:
            self.message_timer -= 1
            if self.message_timer == 0:
                self.message = ""
                if not self.game_over and self.score >= self.target_score:
                    self.level += 1
                    self._setup_level()
        
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _place_element(self):
        # Check if cell is occupied
        for el in self.placed_elements:
            if el['grid_pos'] == self.cursor_pos:
                return 0 # Cannot place on occupied cell

        old_score = self._calculate_aesthetic_score()
        
        element_key = self.available_elements_keys[self.selected_element_index]
        element_def = self.available_elements_defs[element_key]
        
        new_element = {
            "key": element_key,
            "def": element_def,
            "grid_pos": list(self.cursor_pos),
            "anim_offset": self.np_random.uniform(0, 2 * math.pi)
        }
        self.placed_elements.append(new_element)
        
        new_score = self._calculate_aesthetic_score()
        self.score = new_score
        
        # Spawn particles
        # sfx: placement_pop.wav
        screen_pos = self._iso_to_screen(*self.cursor_pos)
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                "pos": list(screen_pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 30),
                "color": random.choice([self.COLOR_PLANT_LEAF, (200,200,200), (255,255,255)])
            })
            
        score_diff = new_score - old_score
        reward = score_diff * 0.1
        if score_diff > 0:
            reward += 1
        return reward

    def _calculate_aesthetic_score(self):
        total_score = 0
        for i in range(len(self.placed_elements)):
            for j in range(i + 1, len(self.placed_elements)):
                el1 = self.placed_elements[i]
                el2 = self.placed_elements[j]
                
                dist = math.hypot(el1['grid_pos'][0] - el2['grid_pos'][0], el1['grid_pos'][1] - el2['grid_pos'][1])
                
                if dist < 1.5: # Clutter penalty
                    total_score -= 10
                    continue

                type1 = el1['def']['type']
                type2 = el2['def']['type']
                
                harmony_bonus = 0
                if type1 == 'rock' and type2 == 'rock': harmony_bonus = 10
                elif type1 == 'plant' and type2 == 'plant': harmony_bonus = 15
                else: harmony_bonus = 25 # Rock-plant harmony is most valued
                
                score_gain = harmony_bonus / (dist**1.5 + 1)
                total_score += score_gain
        
        return max(0, int(total_score))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _iso_to_screen(self, iso_x, iso_y):
        screen_x = self.ORIGIN_X + (iso_x - iso_y) * (self.TILE_W / 2)
        screen_y = self.ORIGIN_Y + (iso_x + iso_y) * (self.TILE_H / 2)
        return int(screen_x), int(screen_y)

    def _draw_iso_poly(self, surface, color, points, offset=(0,0)):
        screen_points = [self._iso_to_screen(p[0] + offset[0], p[1] + offset[1]) for p in points]
        pygame.gfxdraw.filled_polygon(surface, screen_points, color)
        pygame.gfxdraw.aapolygon(surface, screen_points, color)
        
    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_H + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_W, y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for x in range(self.GRID_W + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_H)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

        # Draw cursor
        cursor_poly = [(0,0), (1,0), (1,1), (0,1)]
        self._draw_iso_poly(self.screen, self.COLOR_CURSOR, cursor_poly, self.cursor_pos)
        
        # Sort elements by grid y-position for correct draw order
        sorted_elements = sorted(self.placed_elements, key=lambda el: el['grid_pos'][0] + el['grid_pos'][1])

        # Draw elements
        for el in sorted_elements:
            pos = el['grid_pos']
            el_def = el['def']
            size = el_def['size']
            
            if el_def['type'] == 'rock':
                # Draw rock as a 3D cube
                top_points = [(0,0), (1,0), (1,1), (0,1)]
                left_face = [(0,1), (1,1), (1,2), (0,2)]
                right_face = [(1,0), (1,1), (2,1), (2,0)] # This is not quite right, but simple shapes are fine
                
                center_offset = (1-size)/2
                rock_pos = (pos[0] + center_offset, pos[1] + center_offset)
                
                # Simplified rhombus shape for rock
                face_color_dark = tuple(c*0.8 for c in self.COLOR_ROCK_FACE)
                top_poly = [(0, 0.5), (0.5, 0), (1, 0.5), (0.5, 1)]
                left_poly = [(0, 0.5), (0.5, 1), (0.5, 1.5), (0, 1)]
                right_poly = [(0.5, 1), (1, 0.5), (1, 1), (0.5, 1.5)]
                
                self._draw_iso_poly(self.screen, face_color_dark, [(p[0]*size, p[1]*size) for p in left_poly], rock_pos)
                self._draw_iso_poly(self.screen, self.COLOR_ROCK_FACE, [(p[0]*size, p[1]*size) for p in right_poly], rock_pos)
                self._draw_iso_poly(self.screen, self.COLOR_ROCK_TOP, [(p[0]*size, p[1]*size) for p in top_poly], rock_pos)

            elif el_def['type'] == 'plant':
                # Draw plant
                base_pos = self._iso_to_screen(pos[0] + 0.5, pos[1] + 0.5)
                sway = math.sin(self.steps / 20 + el['anim_offset']) * el_def['sway']
                
                # Stem
                stem_h = 20 * size
                pygame.draw.line(self.screen, self.COLOR_PLANT_STEM, base_pos, (base_pos[0] + sway, base_pos[1] - stem_h), int(4*size))
                
                # Leaves
                leaf_pos = (base_pos[0] + sway, base_pos[1] - stem_h)
                pygame.gfxdraw.filled_circle(self.screen, int(leaf_pos[0]), int(leaf_pos[1]), int(10*size), self.COLOR_PLANT_LEAF)
                pygame.gfxdraw.aacircle(self.screen, int(leaf_pos[0]), int(leaf_pos[1]), int(10*size), self.COLOR_PLANT_LEAF)

        # Draw particles
        for p in self.particles:
            radius = int(p['life'] / 15.0 * 4)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'])
    
    def _render_ui(self):
        # Draw UI Text
        score_surf = self.FONT_UI.render(f"Score: {self.score} / {self.target_score}", True, self.COLOR_TEXT)
        time_surf = self.FONT_UI.render(f"Time: {self.time_remaining // self.FPS:02d}", True, self.COLOR_TEXT)
        level_surf = self.FONT_UI.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        
        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))
        self.screen.blit(level_surf, (self.WIDTH // 2 - level_surf.get_width() // 2, 10))
        
        # Draw selected item preview
        preview_bg = pygame.Rect(self.WIDTH // 2 - 75, self.HEIGHT - 50, 150, 45)
        pygame.draw.rect(self.screen, (*self.COLOR_BG, 220), preview_bg, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, preview_bg, width=1, border_radius=5)
        
        sel_key = self.available_elements_keys[self.selected_element_index]
        sel_name = sel_key.replace("_", " ").title()
        sel_surf = self.FONT_UI.render(sel_name, True, self.COLOR_TEXT)
        self.screen.blit(sel_surf, (preview_bg.centerx - sel_surf.get_width() // 2, preview_bg.centery - sel_surf.get_height() // 2))

        # Draw messages (Game Over, Level Complete)
        if self.message:
            msg_surf = self.FONT_MSG.render(self.message, True, self.COLOR_TEXT)
            bg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            bg_rect.inflate_ip(40, 20)
            
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill(self.COLOR_MSG_BG)
            self.screen.blit(s, bg_rect.topleft)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, bg_rect, 2, 5)
            self.screen.blit(msg_surf, msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "time_remaining": self.time_remaining // self.FPS
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set dummy video driver for headless operation
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    obs, info = env.reset()
    print("Initial Observation Shape:", obs.shape)
    print("Initial Info:", info)

    terminated = False
    total_reward = 0
    for _ in range(3000): # Run for 100 seconds
        action = env.action_space.sample() # Random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print("Episode finished.")
            print(f"Final Info: {info}")
            print(f"Total Reward: {total_reward}")
            obs, info = env.reset()
            total_reward = 0
    
    env.close()
    print("Environment closed.")