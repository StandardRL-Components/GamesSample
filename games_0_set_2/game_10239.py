import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:10:40.284306
# Source Brief: brief_00239.md
# Brief Index: 239
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A puzzle game where you place, transform, and modify shapes to match a target pattern. "
        "Use triggers and catalysts to alter your clones and solve the level."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to place a shape and shift to cycle through catalysts."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.CURSOR_SPEED = 10

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TRIGGER = (100, 80, 200)
        self.COLOR_TARGET_GLOW = (255, 200, 80)
        self.COLOR_SHAPE_1 = (50, 220, 255)  # Circle
        self.COLOR_SHAPE_2 = (255, 80, 100)  # Square
        self.COLOR_SHAPE_3 = (80, 255, 150)  # Triangle

        # Shape and Catalyst definitions
        self.SHAPES = {
            0: {'name': 'CIRCLE', 'color': self.COLOR_SHAPE_1},
            1: {'name': 'SQUARE', 'color': self.COLOR_SHAPE_2},
            2: {'name': 'TRIANGLE', 'color': self.COLOR_SHAPE_3},
        }
        self.CATALYSTS = {
            0: {'name': 'NONE', 'color': (180, 180, 180)},
            1: {'name': 'GROW', 'color': (255, 150, 50)},
            2: {'name': 'SHRINK', 'color': (100, 150, 255)},
        }

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_catalyst = pygame.font.SysFont("Consolas", 24, bold=True)

        # Game state variables are initialized in reset()
        self.cursor_pos = None
        self.cursor_shape_type = None
        self.cursor_size = None
        self.placed_clones = None
        self.triggers = None
        self.target_shapes = None
        self.particles = None
        self.active_catalyst_idx = None
        self.clones_remaining = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.previous_action = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.placed_clones = []
        self.triggers = []
        self.target_shapes = []
        self.particles = []
        self.active_catalyst_idx = 0
        self.previous_action = [0, 0, 0]
        self._generate_level()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        prev_similarity = self._calculate_similarity()
        
        # 1. Handle Input
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and self.previous_action[1] == 0
        shift_pressed = shift_held and self.previous_action[2] == 0
        self.previous_action = action

        self._move_cursor(movement)
        if shift_pressed:
            # SFX: catalyst_cycle.wav
            self.active_catalyst_idx = (self.active_catalyst_idx + 1) % len(self.CATALYSTS)
        if space_pressed:
            self._clone_shape()

        # 2. Update Game Logic
        transformation_reward = self._update_transformations()
        self._update_particles()

        # 3. Calculate Reward
        current_similarity = self._calculate_similarity()
        similarity_reward = (current_similarity - prev_similarity) * 10.0 # Reward for progress
        step_penalty = -0.1 if abs(similarity_reward) < 0.01 and movement > 0 else 0
        
        reward = similarity_reward + transformation_reward + step_penalty
        self.score += reward

        # 4. Check Termination
        terminated = self._check_termination()
        truncated = False # This environment does not truncate based on time limit
        if terminated:
            self.game_over = True
            if self._check_win_condition():
                terminal_reward = 100
            else: # Lost
                terminal_reward = -100
            reward += terminal_reward
            self.score += terminal_reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _generate_level(self):
        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.cursor_shape_type = 0
        self.cursor_size = 20

        level_idx = self.np_random.integers(0, 3)

        if level_idx == 0:
            # Level 1: Place a square on a trigger to make a circle.
            self.clones_remaining = 1
            self.cursor_shape_type = 1 # Square
            self.triggers = [{'pos': pygame.Vector2(320, 200), 'size': 30}]
            self.target_shapes = [{'pos': pygame.Vector2(320, 200), 'type': 0, 'size': 20}] # Circle
        elif level_idx == 1:
            # Level 2: Place a square away, and a triangle on a trigger to make a square.
            self.clones_remaining = 2
            self.cursor_shape_type = 1 # Square
            self.triggers = [{'pos': pygame.Vector2(200, 200), 'size': 30}]
            self.target_shapes = [
                {'pos': pygame.Vector2(200, 200), 'type': 0, 'size': 20}, # Circle
                {'pos': pygame.Vector2(440, 200), 'type': 1, 'size': 20}  # Square
            ]
        else:
            # Level 3: Use the GROW catalyst
            self.clones_remaining = 1
            self.cursor_shape_type = 0 # Circle
            self.triggers = [{'pos': pygame.Vector2(320, 150), 'size': 30}]
            self.target_shapes = [{'pos': pygame.Vector2(320, 150), 'type': 1, 'size': 24}] # Big Square

        self._assign_shape_properties(self.target_shapes)

    def _assign_shape_properties(self, shapes):
        for shape in shapes:
            shape['color'] = self.SHAPES[shape['type']]['color']

    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED # Up
        if movement == 2: self.cursor_pos.y += self.CURSOR_SPEED # Down
        if movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED # Left
        if movement == 4: self.cursor_pos.x += self.CURSOR_SPEED # Right
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

    def _clone_shape(self):
        if self.clones_remaining > 0:
            can_place = True
            for clone in self.placed_clones:
                if clone['pos'].distance_to(self.cursor_pos) < self.cursor_size + clone['size']:
                    can_place = False
                    break
            if can_place:
                # SFX: clone_spawn.wav
                new_clone = {'pos': self.cursor_pos.copy(), 'type': self.cursor_shape_type, 'size': self.cursor_size}
                self._assign_shape_properties([new_clone])
                self.placed_clones.append(new_clone)
                self.clones_remaining -= 1
                self._create_particles(self.cursor_pos, new_clone['color'], 30, 2)
    
    def _update_transformations(self):
        transformation_reward = 0
        transformed_indices = set()

        for i, clone in enumerate(self.placed_clones):
            if i in transformed_indices: continue
            for trigger in self.triggers:
                if clone['pos'].distance_to(trigger['pos']) < clone['size'] / 2 + trigger['size'] / 2:
                    # SFX: transform.wav
                    transformed_indices.add(i)
                    
                    # Apply transformation
                    clone['type'] = (clone['type'] + 1) % len(self.SHAPES)
                    if self.active_catalyst_idx == 1: # GROW
                        clone['size'] = min(40, clone['size'] * 1.2)
                    elif self.active_catalyst_idx == 2: # SHRINK
                        clone['size'] = max(10, clone['size'] * 0.8)
                    
                    self._assign_shape_properties([clone])
                    self._create_particles(clone['pos'], clone['color'], 50, 4)

                    # Check if this transformation was helpful
                    for target in self.target_shapes:
                        if target['type'] == clone['type'] and target['pos'].distance_to(clone['pos']) < 5:
                            transformation_reward += 5
                    break 
        return transformation_reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] = max(0, p['size'] * 0.95)

    def _create_particles(self, pos, color, count, speed_scale):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_scale
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 31),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _check_termination(self):
        is_win = self._check_win_condition()
        is_stalled = self.clones_remaining <= 0 and not self._is_transforming()
        is_timeout = self.steps >= self.MAX_STEPS
        return is_win or (is_stalled and not is_win) or is_timeout

    def _check_win_condition(self):
        return self._calculate_similarity() >= 1.0

    def _is_transforming(self):
        for clone in self.placed_clones:
            for trigger in self.triggers:
                if clone['pos'].distance_to(trigger['pos']) < clone['size'] / 2 + trigger['size'] / 2:
                    return True
        return False

    def _calculate_similarity(self):
        if not self.target_shapes: return 1.0 if not self.placed_clones else 0.0
        
        matches = 0
        used_clone_indices = set()
        
        for target in self.target_shapes:
            best_match_idx = -1
            min_dist_sq = (target['size'])**2 # Match radius

            for i, clone in enumerate(self.placed_clones):
                if i in used_clone_indices: continue
                if clone['type'] == target['type'] and abs(clone['size'] - target['size']) < 3:
                    dist_sq = clone['pos'].distance_squared_to(target['pos'])
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        best_match_idx = i
            
            if best_match_idx != -1:
                matches += 1
                used_clone_indices.add(best_match_idx)
                
        return matches / len(self.target_shapes)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "clones_remaining": self.clones_remaining, "similarity": self._calculate_similarity()}

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        
        # Draw target shape outlines
        for shape in self.target_shapes:
            self._draw_glowing_shape(self.screen, shape['type'], shape['pos'], shape['size'], self.COLOR_TARGET_GLOW, is_outline=True)
            
        # Draw triggers
        for trigger in self.triggers:
            glow_color = (*self.COLOR_TRIGGER, 50)
            pygame.gfxdraw.filled_circle(self.screen, int(trigger['pos'].x), int(trigger['pos'].y), int(trigger['size']), (*self.COLOR_TRIGGER, 100))
            pygame.gfxdraw.aacircle(self.screen, int(trigger['pos'].x), int(trigger['pos'].y), int(trigger['size']), self.COLOR_TRIGGER)
            for i in range(5):
                 pygame.gfxdraw.aacircle(self.screen, int(trigger['pos'].x), int(trigger['pos'].y), int(trigger['size'] + i*2), (*glow_color[:3], 50-i*10))

        # Draw placed clones
        for clone in self.placed_clones:
            self._draw_glowing_shape(self.screen, clone['type'], clone['pos'], clone['size'], clone['color'])

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            if alpha > 0:
                color = (*p['color'], alpha)
                pos = (int(p['pos'].x), int(p['pos'].y))
                size = int(p['size'])
                if size > 0:
                    rect = pygame.Rect(pos[0] - size // 2, pos[1] - size // 2, size, size)
                    shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                    pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
                    self.screen.blit(shape_surf, rect.topleft)

        # Draw cursor
        if not self.game_over:
            cursor_color = self.SHAPES[self.cursor_shape_type]['color']
            self._draw_glowing_shape(self.screen, self.cursor_shape_type, self.cursor_pos, self.cursor_size, cursor_color, glow_color=self.COLOR_CURSOR)

    def _draw_glowing_shape(self, surface, shape_type, pos, size, color, glow_color=None, is_outline=False):
        glow_color = glow_color if glow_color else color
        int_pos = (int(pos.x), int(pos.y))
        int_size = int(size)

        # Glow effect
        for i in range(4, 0, -1):
            glow_alpha = 40 if not is_outline else 60
            current_glow_color = (*glow_color[:3], int(glow_alpha / (i / 2.0)))
            self._draw_primitive(surface, shape_type, int_pos, int_size + i * 2, current_glow_color, filled=True)
        
        # Main shape
        if is_outline:
            self._draw_primitive(surface, shape_type, int_pos, int_size, color, filled=False, line_width=2)
        else:
            self._draw_primitive(surface, shape_type, int_pos, int_size, color, filled=True)

    def _draw_primitive(self, surface, shape_type, pos, size, color, filled=True, line_width=1):
        if size <= 0: return
        if shape_type == 0: # Circle
            if filled:
                pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], size, color)
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], size, color)
            if not filled and line_width > 1:
                 pygame.gfxdraw.aacircle(surface, pos[0], pos[1], size-1, color)
        elif shape_type == 1: # Square
            rect = pygame.Rect(pos[0] - size, pos[1] - size, size * 2, size * 2)
            if filled:
                pygame.draw.rect(surface, color, rect, border_radius=3)
            else:
                pygame.draw.rect(surface, color, rect, width=line_width, border_radius=3)
        elif shape_type == 2: # Triangle
            points = [
                (pos[0], pos[1] - size * 1.1),
                (pos[0] - size, pos[1] + size * 0.7),
                (pos[0] + size, pos[1] + size * 0.7),
            ]
            if filled:
                pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, color)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        # Clones
        clones_text = self.font_main.render(f"CLONES: {self.clones_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(clones_text, (10, 10))
        # Catalyst
        catalyst = self.CATALYSTS[self.active_catalyst_idx]
        catalyst_text = self.font_catalyst.render(f"CATALYST: {catalyst['name']}", True, catalyst['color'])
        self.screen.blit(catalyst_text, (self.WIDTH/2 - catalyst_text.get_width()/2, self.HEIGHT - 35))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game with keyboard controls
    # It will not be run by the evaluator
    
    # Un-comment the line below to run with a visible display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Shape Cloner")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Replace the main loop to use the new step API
    while running:
        # --- Human Controls ---
        movement = 0 # No-op
        space_held = False
        shift_held = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        pygame.surfarray.blit_array(screen, np.transpose(obs, (1, 0, 2)))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            pygame.time.wait(2000) # Pause before reset
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()
    pygame.quit()