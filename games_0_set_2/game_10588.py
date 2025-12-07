import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:37:49.351400
# Source Brief: brief_00588.md
# Brief Index: 588
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a puzzle game where the goal is to merge three lights.

    **Game Concept:**
    The player controls three lights (red, green, blue) on a 7x7 grid. The objective is
    to move them onto the same square to merge them. As lights merge, they become
    brighter, larger, and move faster across the grid.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `action[0]` (Movement): 0=None, 1=Up, 2=Down, 3=Left, 4=Right
    - `action[1]` (Select): 1=Press to cycle through active lights
    - `action[2]` (Unused): No function

    **Observation Space:** `Box(0, 255, (400, 640, 3), uint8)`
    - An RGB image of the game screen.

    **Rewards:**
    - +1.0 for moving lights closer together.
    - -0.1 for moving lights farther apart.
    - +10.0 for each successful merge of two lights.
    - +100.0 for the final merge (victory).

    **Termination:**
    - The episode ends when all three lights are merged into one.
    - The episode ends if the maximum step count (1000) is reached.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = "Control three colored lights on a grid. Move them onto the same square to merge them and win."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move the selected light. Press space to cycle which light is selected."
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 7
    GRID_PADDING = 40
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (40, 60, 90)
    COLOR_TEXT = (220, 220, 240)
    
    # Light Colors mapping based on the set of original light IDs (0, 1, 2)
    LIGHT_COLORS = {
        frozenset({0}): (255, 50, 50),       # Red
        frozenset({1}): (50, 255, 50),       # Green
        frozenset({2}): (50, 100, 255),      # Blue
        frozenset({0, 1}): (255, 255, 50),   # Yellow (R+G)
        frozenset({0, 2}): (255, 50, 255),   # Magenta (R+B)
        frozenset({1, 2}): (50, 255, 255),   # Cyan (G+B)
        frozenset({0, 1, 2}): (255, 255, 255), # White (R+G+B)
    }
    
    # --- Initialization ---
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.cell_size = min(
            (self.SCREEN_WIDTH - 2 * self.GRID_PADDING) // self.GRID_SIZE,
            (self.SCREEN_HEIGHT - 2 * self.GRID_PADDING) // self.GRID_SIZE
        )
        self.grid_width = self.cell_size * self.GRID_SIZE
        self.grid_height = self.cell_size * self.GRID_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lights = []
        self.selected_light_idx = 0
        self.space_was_held = False
        self.particles = []
        self.stars = []
        
    # --- Gym Interface ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.space_was_held = False
        self.particles = []
        
        self.stars = [
            (
                (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)),
                self.np_random.uniform(0.5, 1.5),
                self.np_random.integers(50, 100)
            ) for _ in range(100)
        ]

        self.lights = []
        occupied_positions = set()
        for i in range(3):
            while True:
                pos = (self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE))
                if pos not in occupied_positions:
                    occupied_positions.add(pos)
                    break
            
            self.lights.append({
                'pos': pygame.Vector2(pos),
                'anim_start_pos': pygame.Vector2(pos),
                'anim_progress': 1.0,
                'speed': 1,
                'merged_ids': frozenset({i}),
            })
            
        self.selected_light_idx = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        space_pressed = space_held and not self.space_was_held
        if space_pressed and self.lights:
            self.selected_light_idx = (self.selected_light_idx + 1) % len(self.lights)
            # Sound placeholder: play_select_sound()
        self.space_was_held = space_held

        if movement > 0 and self.lights:
            light = self.lights[self.selected_light_idx]
            dist_before = self._get_total_distance()

            delta = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            new_pos = light['pos'] + pygame.Vector2(delta) * light['speed']
            
            new_pos.x = max(0, min(self.GRID_SIZE - 1, new_pos.x))
            new_pos.y = max(0, min(self.GRID_SIZE - 1, new_pos.y))
            
            if new_pos != light['pos']:
                light['anim_start_pos'] = pygame.Vector2(light['pos'])
                light['pos'] = new_pos
                light['anim_progress'] = 0.0
                # Sound placeholder: play_move_sound()

            dist_after = self._get_total_distance()
            if dist_after < dist_before:
                reward += 1.0
            elif dist_after > dist_before:
                reward -= 0.1

        merged_this_step = True
        while merged_this_step:
            merged_this_step, merge_reward = self._check_and_perform_merge()
            if merged_this_step:
                reward = merge_reward # Merge reward overrides movement reward

        self.steps += 1
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and len(self.lights) == 1 and len(self.lights[0]['merged_ids']) == 3:
             reward += 100
        
        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Private Helper Methods ---
    def _check_and_perform_merge(self):
        if len(self.lights) <= 1:
            return False, 0
        
        for i in range(len(self.lights)):
            for j in range(i + 1, len(self.lights)):
                light1 = self.lights[i]
                light2 = self.lights[j]

                if light1['pos'] == light2['pos']:
                    new_merged_ids = light1['merged_ids'].union(light2['merged_ids'])
                    new_light = {
                        'pos': pygame.Vector2(light1['pos']),
                        'anim_start_pos': pygame.Vector2(light1['pos']),
                        'anim_progress': 1.0,
                        'speed': len(new_merged_ids),
                        'merged_ids': new_merged_ids,
                    }

                    color = self.LIGHT_COLORS.get(new_merged_ids, (255,255,255))
                    pixel_pos = self._grid_to_pixel(new_light['pos'])
                    self._spawn_particles(pixel_pos, color, 50)
                    # Sound placeholder: play_merge_sound()

                    self.lights.pop(j)
                    self.lights.pop(i)
                    self.lights.append(new_light)
                    self.selected_light_idx = min(self.selected_light_idx, len(self.lights) - 1)
                    
                    return True, 10.0
        
        return False, 0

    def _get_total_distance(self):
        if len(self.lights) < 2: return 0
        total_dist = 0
        for i in range(len(self.lights)):
            for j in range(i + 1, len(self.lights)):
                total_dist += self.lights[i]['pos'].distance_to(self.lights[j]['pos'])
        return total_dist

    def _check_termination(self):
        if len(self.lights) == 1 and len(self.lights[0]['merged_ids']) == 3:
            # Sound placeholder: play_victory_sound()
            return True
        # The truncation condition is handled in step()
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lights_remaining": len(self.lights),
        }

    # --- Rendering Methods ---
    def _render_game(self):
        for pos, radius, brightness in self.stars:
            color = (brightness, brightness, int(brightness*1.2))
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(radius), color)

        for i in range(self.GRID_SIZE + 1):
            start_x = self.grid_offset_x + i * self.cell_size
            pygame.draw.aaline(self.screen, self.COLOR_GRID, (start_x, self.grid_offset_y), (start_x, self.grid_offset_y + self.grid_height))
            start_y = self.grid_offset_y + i * self.cell_size
            pygame.draw.aaline(self.screen, self.COLOR_GRID, (self.grid_offset_x, start_y), (self.grid_offset_x + self.grid_width, start_y))

        self._update_and_render_particles()

        for i, light in enumerate(self.lights):
            if light['anim_progress'] < 1.0:
                light['anim_progress'] = min(1.0, light['anim_progress'] + 0.1)
            
            lerp_pos = light['anim_start_pos'].lerp(light['pos'], light['anim_progress'])
            pixel_pos = self._grid_to_pixel(lerp_pos)
            
            num_merged = len(light['merged_ids'])
            radius = int(self.cell_size * 0.25 + (num_merged - 1) * 5)
            color = self.LIGHT_COLORS.get(light['merged_ids'], (255,255,255))
            
            self._draw_glowing_circle(self.screen, color, pixel_pos, radius)

            if i == self.selected_light_idx and not self.game_over:
                pulse = abs(math.sin(pygame.time.get_ticks() * 0.005))
                select_color = (255, 255, 255, 100 + pulse * 100)
                select_radius = radius + 5 + pulse * 3
                
                temp_surf = pygame.Surface((int(select_radius*2), int(select_radius*2)), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, select_color, (select_radius, select_radius), select_radius, width=3)
                self.screen.blit(temp_surf, (int(pixel_pos.x - select_radius), int(pixel_pos.y - select_radius)))
                
    def _render_ui(self):
        goal_text = "Goal: Merge all lights into one"
        if self.game_over and len(self.lights) == 1 and len(self.lights[0]['merged_ids']) == 3:
            goal_text = "VICTORY!"
        elif self.game_over:
            goal_text = "GAME OVER"
            
        text_surface = self.font_large.render(goal_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH // 2 - text_surface.get_width() // 2, 10))
        
        steps_surf = self.font_small.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        lights_surf = self.font_small.render(f"Lights: {len(self.lights)}", True, self.COLOR_TEXT)
        self.screen.blit(steps_surf, (10, 10))
        self.screen.blit(lights_surf, (10, 30))
        
        if self.lights and not self.game_over:
            light = self.lights[self.selected_light_idx]
            color = self.LIGHT_COLORS.get(light['merged_ids'], (255,255,255))
            
            selected_surf = self.font_small.render(f"Selected: Speed {light['speed']}x", True, color)
            self.screen.blit(selected_surf, (self.SCREEN_WIDTH - selected_surf.get_width() - 10, 10))

    def _grid_to_pixel(self, grid_pos):
        x = self.grid_offset_x + grid_pos.x * self.cell_size + self.cell_size / 2
        y = self.grid_offset_y + grid_pos.y * self.cell_size + self.cell_size / 2
        return pygame.Vector2(x, y)

    def _draw_glowing_circle(self, surface, color, pos, radius):
        for i in range(4):
            glow_radius = int(radius + i * 4)
            glow_alpha = 80 - i * 20
            
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                temp_surf, (*color, glow_alpha), (glow_radius, glow_radius), glow_radius
            )
            surface.blit(temp_surf, (int(pos.x - glow_radius), int(pos.y - glow_radius)))
            
        pygame.gfxdraw.filled_circle(surface, int(pos.x), int(pos.y), int(radius), color)
        pygame.gfxdraw.aacircle(surface, int(pos.x), int(pos.y), int(radius), color)

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos), 'vel': velocity, 'lifespan': self.np_random.integers(20, 41),
                'color': color, 'radius': self.np_random.uniform(1, 4)
            })

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] *= 0.97
            
            if p['lifespan'] <= 0 or p['radius'] < 0.5:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['lifespan'] / 40))
                temp_surf = pygame.Surface((int(p['radius']*2), int(p['radius']*2)), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*p['color'], alpha), (p['radius'], p['radius']), p['radius'])
                self.screen.blit(temp_surf, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # The main loop is for human play and is not part of the environment tests
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.init()
    pygame.font.init()
    
    pygame.display.set_caption("Merge Lights Gym Environment")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n--- Human Controls ---")
    print("Arrows: Move selected light")
    print("Space:  Select next light (on key press)")
    print("R:      Reset environment")
    print("Q/ESC:  Quit")
    
    # For detecting key presses vs holds
    key_action_map = {
        pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4
    }
    space_pressed_in_frame = False

    while running:
        movement = 0
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Environment Reset ---")
                if event.key == pygame.K_SPACE:
                    space_pressed_in_frame = True
        
        # Check for held keys for movement
        keys = pygame.key.get_pressed()
        for key, move_action in key_action_map.items():
            if keys[key]:
                movement = move_action
                break

        if space_pressed_in_frame:
            space = 1
        
        # Take a step
        action = [movement, space, 0]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        space_pressed_in_frame = False
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30)
        
    pygame.quit()