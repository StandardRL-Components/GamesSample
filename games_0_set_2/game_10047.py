import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:50:33.892618
# Source Brief: brief_00047.md
# Brief Index: 47
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import copy

class GameEnv(gym.Env):
    """
    A Gymnasium environment for "Sandman's Workshop".

    The player controls a cursor to gather dream materials, craft potions,
    and evade a dream predator. The goal is to craft the ultimate slumber
    potion before being caught or running out of time. The game features
    a time-reversal mechanic.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Gather dream materials, craft potions, and evade a dream predator. "
        "Craft the ultimate slumber potion to win before being caught or running out of time."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to collect materials or craft potions. "
        "Press shift to reverse time."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    UI_WIDTH = 180
    GAME_WIDTH = SCREEN_WIDTH - UI_WIDTH

    # Colors (Dreamlike Palette)
    COLOR_BG = (10, 5, 25)
    COLOR_UI_BG = (20, 15, 40)
    COLOR_CURSOR = (255, 255, 100)
    COLOR_CURSOR_GLOW = (255, 255, 0, 50)
    COLOR_PREDATOR = (200, 20, 50)
    COLOR_PREDATOR_GLOW = (255, 20, 50, 60)
    COLOR_TIME_REVERSAL = (180, 100, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_DIM = (100, 100, 120)
    COLOR_TEXT_SUCCESS = (100, 255, 100)
    COLOR_TEXT_FAIL = (255, 100, 100)

    MATERIALS = {
        "Stardust": {"color": (255, 255, 180), "glow": (255, 255, 180, 50)},
        "Moonbeam": {"color": (180, 220, 255), "glow": (180, 220, 255, 50)},
        "Whisper": {"color": (220, 180, 255), "glow": (220, 180, 255, 50)},
    }

    RECIPES = {
        "Potion of Calm": {"Stardust": 2, "Whisper": 1},
        "Vision Dust": {"Stardust": 1, "Moonbeam": 2},
        "Ultimate Slumber Potion": {"Stardust": 3, "Moonbeam": 3, "Whisper": 3},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 20)
        self.font_m = pygame.font.Font(None, 24)
        self.font_l = pygame.font.Font(None, 32)
        
        # State variables are initialized in reset()
        self.cursor_pos = None
        self.inventory = None
        self.materials_on_screen = None
        self.predator = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.crafted_potions = None
        self.state_history = None
        self.time_reversal_effect = None
        self.last_reward_info = None

        # The reset method is called here to initialize np_random,
        # but the full state will be set up again in the user's first call to reset()
        self.reset()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Game State
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = np.array([self.GAME_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.inventory = {mat: 0 for mat in self.MATERIALS}
        self.crafted_potions = set()
        self.last_reward_info = {"text": "Begin your craft...", "color": self.COLOR_TEXT, "timer": 100}

        # Entities
        self._spawn_initial_materials()
        self.predator = self._create_predator()
        
        # Effects
        self.particles = []
        self.time_reversal_effect = 0

        # Mechanics
        self.state_history = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Store state for time reversal ---
        # Deepcopy is crucial to prevent modifications affecting the history
        current_state = self._get_game_state()
        self.state_history.append(current_state)
        if len(self.state_history) > 50: # Limit history size
            self.state_history.pop(0)

        # --- Unpack and process actions ---
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.1 # Small penalty for each step
        self.steps += 1
        
        # Action: Time Reversal (Shift)
        if shift_press and len(self.state_history) > 1:
            # Pop current state, then pop previous state to load
            self.state_history.pop() 
            prev_state = self.state_history.pop()
            self._set_game_state(prev_state)
            self.time_reversal_effect = 15 # Visual effect counter
            # SFX: Time rewind sound
            self._add_reward_info("Time Reversed!", self.COLOR_TIME_REVERSAL, 0)

        # Action: Movement
        self._handle_movement(movement)

        # Action: Interaction (Space)
        interaction_reward = self._handle_interaction(space_press)
        reward += interaction_reward

        # --- Update Game World ---
        self._update_materials()
        self._update_predator()
        self._update_particles()
        if self.time_reversal_effect > 0:
            self.time_reversal_effect -= 1
        if self.last_reward_info['timer'] > 0:
            self.last_reward_info['timer'] -= 1

        # --- Check for Termination ---
        terminated = False
        terminal_reward = self._check_termination()
        if terminal_reward != 0:
            reward += terminal_reward
            terminated = True
            self.game_over = True
        
        if self.steps >= 1000:
            # This is a truncation condition, not termination
            truncated = True
            self.game_over = True
            return self._get_observation(), reward, False, truncated, self._get_info()

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Core Logic Sub-functions ---

    def _handle_movement(self, movement):
        cursor_speed = 8.0
        if movement == 1: # Up
            self.cursor_pos[1] -= cursor_speed
        elif movement == 2: # Down
            self.cursor_pos[1] += cursor_speed
        elif movement == 3: # Left
            self.cursor_pos[0] -= cursor_speed
        elif movement == 4: # Right
            self.cursor_pos[0] += cursor_speed
        
        # Clamp cursor position
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT - 1)

    def _handle_interaction(self, space_press):
        if not space_press:
            return 0
        
        reward = 0
        # Check for material collection
        for mat in self.materials_on_screen[:]:
            dist = np.linalg.norm(self.cursor_pos - mat['pos'])
            if dist < mat['size'] + 5: # 5 is cursor radius
                self.inventory[mat['name']] += 1
                self.materials_on_screen.remove(mat)
                self._spawn_particle_burst(mat['pos'], self.MATERIALS[mat['name']]['color'], 20)
                reward += 1.0
                self._add_reward_info(f"+1 {mat['name']}", self.MATERIALS[mat['name']]['color'], 1.0)
                # SFX: Material collect chime
                return reward # Interact with only one thing per step

        # Check for crafting interaction
        if self.cursor_pos[0] > self.GAME_WIDTH:
            for i, (name, recipe) in enumerate(self.RECIPES.items()):
                button_y = 180 + i * 65 # Updated to match rendering
                button_rect = pygame.Rect(self.GAME_WIDTH + 10, button_y, self.UI_WIDTH - 20, 50)
                if button_rect.collidepoint(self.cursor_pos):
                    if self._can_craft(recipe):
                        self._craft_item(name, recipe)
                        # SFX: Potion craft success
                        if name not in self.crafted_potions:
                            self.crafted_potions.add(name)
                            reward += 5.0
                            self._add_reward_info(f"New Potion!", self.COLOR_TEXT_SUCCESS, 5.0)
                        else:
                            self._add_reward_info(f"Crafted!", self.COLOR_TEXT, 0)
                        
                        if name == "Ultimate Slumber Potion":
                            # This is handled in termination check, but we can set a flag
                            pass
                    else:
                        # SFX: Crafting fail buzzer
                        self._add_reward_info(f"Need Materials", self.COLOR_TEXT_FAIL, 0)
        return reward

    def _check_termination(self):
        # Victory Condition
        if "Ultimate Slumber Potion" in self.crafted_potions:
            self._add_reward_info("VICTORY!", self.COLOR_TEXT_SUCCESS, 100)
            return 100

        # Failure Condition
        predator_dist = np.linalg.norm(self.cursor_pos - self.predator['pos'])
        if predator_dist < 20: # 5 for cursor, 15 for predator
            self._add_reward_info("CAUGHT!", self.COLOR_PREDATOR, -100)
            return -100
        
        return 0

    def _update_predator(self):
        # Speed scaling
        self.predator['speed'] = 1.0 + (self.steps // 200) * 0.05

        target_pos = self.predator['path'][self.predator['path_idx']]
        direction = target_pos - self.predator['pos']
        dist = np.linalg.norm(direction)

        if dist < self.predator['speed']:
            self.predator['pos'] = target_pos
            self.predator['path_idx'] = (self.predator['path_idx'] + 1) % len(self.predator['path'])
        else:
            self.predator['pos'] += (direction / dist) * self.predator['speed']

    def _update_materials(self):
        # Gently float materials
        for mat in self.materials_on_screen:
            mat['vel'] += self.np_random.uniform(-0.02, 0.02, 2)
            mat['vel'] = np.clip(mat['vel'], -0.2, 0.2)
            mat['pos'] += mat['vel']
            mat['pos'][0] = np.clip(mat['pos'][0], 10, self.GAME_WIDTH - 10)
            mat['pos'][1] = np.clip(mat['pos'][1], 10, self.SCREEN_HEIGHT - 10)
        
        # Respawn if too few
        if len(self.materials_on_screen) < 8:
            self._spawn_material()

    # --- State Management ---
    
    def _get_game_state(self):
        return {
            "steps": self.steps,
            "score": self.score,
            "cursor_pos": self.cursor_pos.copy(),
            "inventory": self.inventory.copy(),
            "crafted_potions": self.crafted_potions.copy(),
            "materials_on_screen": copy.deepcopy(self.materials_on_screen),
            "predator": copy.deepcopy(self.predator),
            "last_reward_info": self.last_reward_info.copy(),
        }

    def _set_game_state(self, state):
        self.steps = state["steps"]
        self.score = state["score"]
        self.cursor_pos = state["cursor_pos"]
        self.inventory = state["inventory"]
        self.crafted_potions = state["crafted_potions"]
        self.materials_on_screen = state["materials_on_screen"]
        self.predator = state["predator"]
        self.last_reward_info = state["last_reward_info"]

    # --- Spawning and Creation ---

    def _spawn_initial_materials(self):
        self.materials_on_screen = []
        for _ in range(12):
            self._spawn_material()

    def _spawn_material(self):
        mat_name = self.np_random.choice(list(self.MATERIALS.keys()))
        self.materials_on_screen.append({
            "name": mat_name,
            "pos": np.array([
                self.np_random.uniform(20, self.GAME_WIDTH - 20),
                self.np_random.uniform(20, self.SCREEN_HEIGHT - 20)
            ]),
            "vel": self.np_random.uniform(-0.1, 0.1, 2).astype(np.float32),
            "size": self.np_random.uniform(8, 12),
        })

    def _create_predator(self):
        path = np.array([
            [50, 50], [self.GAME_WIDTH - 50, 50],
            [self.GAME_WIDTH - 50, self.SCREEN_HEIGHT - 50],
            [50, self.SCREEN_HEIGHT - 50]
        ], dtype=np.float32)
        return {
            'pos': path[0].copy(),
            'path': path,
            'path_idx': 0,
            'speed': 1.0,
            'size': 15
        }
    
    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game_area()
        self._render_ui()
        
        # Apply visual effects
        if self.time_reversal_effect > 0:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(self.time_reversal_effect / 15 * 100)
            overlay.fill((*self.COLOR_TIME_REVERSAL, alpha))
            self.screen.blit(overlay, (0, 0))

        if not self.game_over:
            self._render_predator_proximity_warning()

        self._render_cursor()

        if self.game_over:
            self._render_game_over_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game_area(self):
        # Draw materials
        for mat in self.materials_on_screen:
            info = self.MATERIALS[mat['name']]
            self._draw_glow_circle(self.screen, mat['pos'], info['glow'], mat['size'] * 2.5)
            pygame.gfxdraw.filled_circle(self.screen, int(mat['pos'][0]), int(mat['pos'][1]), int(mat['size']), info['color'])
            pygame.gfxdraw.aacircle(self.screen, int(mat['pos'][0]), int(mat['pos'][1]), int(mat['size']), info['color'])

        # Draw predator
        p = self.predator
        self._draw_glow_circle(self.screen, p['pos'], self.COLOR_PREDATOR_GLOW, p['size'] * 3)
        pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), self.COLOR_PREDATOR)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], p['size'])

    def _render_ui(self):
        # UI Background
        ui_rect = pygame.Rect(self.GAME_WIDTH, 0, self.UI_WIDTH, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_TEXT_DIM, (self.GAME_WIDTH, 0), (self.GAME_WIDTH, self.SCREEN_HEIGHT), 2)
        
        # Title
        title_surf = self.font_l.render("Workshop", True, self.COLOR_TEXT)
        self.screen.blit(title_surf, (self.GAME_WIDTH + (self.UI_WIDTH - title_surf.get_width()) / 2, 10))

        # Inventory
        inv_title_surf = self.font_m.render("Inventory", True, self.COLOR_TEXT)
        self.screen.blit(inv_title_surf, (self.GAME_WIDTH + 10, 50))
        for i, (name, count) in enumerate(self.inventory.items()):
            color = self.MATERIALS[name]['color']
            text = f"{name}: {count}"
            inv_surf = self.font_s.render(text, True, color)
            self.screen.blit(inv_surf, (self.GAME_WIDTH + 20, 80 + i * 20))

        # Crafting
        craft_title_surf = self.font_m.render("Crafting", True, self.COLOR_TEXT)
        self.screen.blit(craft_title_surf, (self.GAME_WIDTH + 10, 150))
        for i, (name, recipe) in enumerate(self.RECIPES.items()):
            y_pos = 180 + i * 65
            can_craft = self._can_craft(recipe)
            color = self.COLOR_TEXT_SUCCESS if can_craft else self.COLOR_TEXT_FAIL
            
            recipe_surf = self.font_m.render(name, True, color)
            self.screen.blit(recipe_surf, (self.GAME_WIDTH + 20, y_pos))

            req_text = ", ".join([f"{v} {k[0]}" for k, v in recipe.items()])
            req_surf = self.font_s.render(req_text, True, self.COLOR_TEXT_DIM)
            self.screen.blit(req_surf, (self.GAME_WIDTH + 25, y_pos + 25))

    def _render_cursor(self):
        pos = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        self._draw_glow_circle(self.screen, pos, self.COLOR_CURSOR_GLOW, 20)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_CURSOR)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, self.COLOR_CURSOR)

    def _render_predator_proximity_warning(self):
        dist = np.linalg.norm(self.cursor_pos - self.predator['pos'])
        max_dist = 200
        if dist < max_dist:
            alpha = int((1 - dist / max_dist) ** 2 * 100)
            warning_color = (*self.COLOR_PREDATOR, alpha)
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(s, warning_color, (0, 0, self.GAME_WIDTH, 5)) # Top
            pygame.draw.rect(s, warning_color, (0, self.SCREEN_HEIGHT-5, self.GAME_WIDTH, 5)) # Bottom
            pygame.draw.rect(s, warning_color, (0, 0, 5, self.SCREEN_HEIGHT)) # Left
            pygame.draw.rect(s, warning_color, (self.GAME_WIDTH-5, 0, 5, self.SCREEN_HEIGHT)) # Right of game area
            self.screen.blit(s, (0, 0))

    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text = self.last_reward_info['text']
        color = self.last_reward_info['color']
        
        end_surf = self.font_l.render(text, True, color)
        pos = ((self.SCREEN_WIDTH - end_surf.get_width()) // 2, (self.SCREEN_HEIGHT - end_surf.get_height()) // 2)
        self.screen.blit(end_surf, pos)
        
        score_text = f"Final Score: {self.score:.1f}"
        score_surf = self.font_m.render(score_text, True, self.COLOR_TEXT)
        score_pos = ((self.SCREEN_WIDTH - score_surf.get_width()) // 2, pos[1] + 40)
        self.screen.blit(score_surf, score_pos)

    # --- Helper & Utility Functions ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "inventory": self.inventory,
            "crafted_potions": len(self.crafted_potions),
            "time_reversals_left": len(self.state_history)
        }

    def _can_craft(self, recipe):
        return all(self.inventory[mat] >= count for mat, count in recipe.items())

    def _craft_item(self, name, recipe):
        if not self._can_craft(recipe): return
        for mat, count in recipe.items():
            self.inventory[mat] -= count
        self._spawn_particle_burst(self.cursor_pos, self.COLOR_TIME_REVERSAL, 50)

    def _add_reward_info(self, text, color, reward_val):
        # This is purely for visual feedback, not part of the RL info dict
        self.last_reward_info = {"text": text, "color": color, "timer": 60}

    @staticmethod
    def _draw_glow_circle(surface, pos, color, radius):
        """Draws a soft, glowing circle. `color` should have alpha."""
        pos = (int(pos[0]), int(pos[1]))
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, color, (radius, radius), radius)
        surface.blit(surf, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _spawn_particle_burst(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'size': self.np_random.uniform(2, 5),
                'color': color,
                'life': self.np_random.integers(20, 40)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['life'] -= 1
            p['size'] *= 0.97
            if p['life'] <= 0 or p['size'] < 0.5:
                self.particles.remove(p)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Sandman's Workshop")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    # --- Manual Control Mapping ---
    # ARROWS: Move cursor
    # SPACE: Interact
    # LSHIFT: Time Reverse
    
    while not terminated and not truncated:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term
        truncated = trunc

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()