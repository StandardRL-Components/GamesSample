import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:14:05.795865
# Source Brief: brief_00883.md
# Brief Index: 883
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import defaultdict

class GameEnv(gym.Env):
    """
    Gymnasium environment for a "Suika-like" shape-merging puzzle game.

    **Objective:** Merge falling circles of the same level to create larger circles
    of the next level. Reach level 15 to win.

    **Game Over:** The game ends if the play area fills up, preventing new
    circles from spawning.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `action[0]` (Movement):
        - 0: None
        - 1: None (was Up)
        - 2: Fast Drop (was Down)
        - 3: Move Left
        - 4: Move Right
    - `action[1]` (Space): Unused
    - `action[2]` (Shift): Unused

    **Observation Space:** A 640x400 RGB image of the game state.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Drop and merge circles of the same size to create larger ones. "
        "Score points by merging and try to create the largest circle without overflowing the container."
    )
    user_guide = (
        "Controls: Use ← and → to move the falling circle, and ↓ to drop it faster."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_LEVEL = 15
        self.WIN_LEVEL = 15
        self.MAX_STEPS = 10000
        self.GRID_SIZE = 20
        self.GRAVITY = 0.15
        self.FRICTION = 0.98
        self.WALL_BOUNCINESS = 0.5
        self.SHAPE_BOUNCINESS = 0.3
        self.PLAYER_FORCE = 0.6
        self.FAST_DROP_FORCE = 0.5
        self.SETTLE_VEL_THRESHOLD = 0.1
        self.SETTLE_FRAMES_REQ = 15
        self.DANGER_ZONE_Y = 60

        # --- Colors & Style ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_DANGER = (255, 0, 0, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_BG = (30, 40, 60, 180)
        self.COLOR_PREVIEW_OUTLINE = (100, 110, 130)
        self.SHAPE_COLORS = self._generate_color_palette(self.MAX_LEVEL + 1)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)

        # --- Game State (initialized in reset) ---
        self.steps = None
        self.score = None
        self.level = None
        self.game_over = None
        self.shapes = None
        self.falling_shape_idx = None
        self.next_shape_level = None
        self.particles = None
        self.level_up_score_target = None
        self.fall_speed_multiplier = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self.shapes = []
        self.falling_shape_idx = None
        self.particles = []
        self._update_level_vars()

        # Determine the first two shapes
        self.next_shape_level = self.np_random.integers(1, 4)
        self._spawn_new_shape() # This sets self.falling_shape_idx and prepares the next one

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        step_reward = 0

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Game Logic Update ---
        self._apply_player_input(movement)
        self._update_physics()
        merge_reward = self._handle_merges()
        step_reward += merge_reward

        # --- State Progression ---
        if self.falling_shape_idx is None and not self.game_over:
            if self._spawn_new_shape():
                self.game_over = True # Spawning failed, game over
                step_reward -= 100 # Loss penalty
            # sfx: new_shape_spawn.wav

        level_up_reward = self._check_level_up()
        step_reward += level_up_reward

        self._update_particles()
        
        # --- Termination Check ---
        terminated = self.game_over
        if not terminated and self.level >= self.WIN_LEVEL:
            terminated = True
            step_reward += 100 # Win bonus
        if not terminated and self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_level_vars(self):
        self.level_up_score_target = int(10 * (1.8 ** (self.level - 1)))
        self.fall_speed_multiplier = 1.0 + (self.level - 1) * 0.05

    def _get_shape_props(self, level):
        radius = 8 + level * 2
        color = self.SHAPE_COLORS[level]
        return radius, color

    def _spawn_new_shape(self):
        """Spawns the next shape. Returns True if spawn fails (game over)."""
        spawn_level = self.next_shape_level
        self.next_shape_level = self.np_random.integers(1, min(4, self.level + 1))
        
        radius, color = self._get_shape_props(spawn_level)
        spawn_pos = [self.WIDTH / 2, self.DANGER_ZONE_Y - radius - 5]
        
        # Game over check: if spawn area is obstructed
        for other in self.shapes:
            if not other['is_falling']:
                dist = math.hypot(spawn_pos[0] - other['pos'][0], spawn_pos[1] - other['pos'][1])
                if dist < radius + other['radius']:
                    # sfx: game_over.wav
                    return True # Failure

        new_shape = {
            'pos': spawn_pos,
            'vel': [0, 0],
            'level': spawn_level,
            'radius': radius,
            'color': color,
            'is_falling': True,
            'settle_timer': 0,
        }
        self.shapes.append(new_shape)
        self.falling_shape_idx = len(self.shapes) - 1
        return False # Success

    def _apply_player_input(self, movement):
        if self.falling_shape_idx is not None:
            shape = self.shapes[self.falling_shape_idx]
            if movement == 3: # Left
                shape['vel'][0] -= self.PLAYER_FORCE
            elif movement == 4: # Right
                shape['vel'][0] += self.PLAYER_FORCE
            elif movement == 2: # Fast Drop
                shape['vel'][1] += self.FAST_DROP_FORCE

    def _update_physics(self):
        # Multiple sub-steps for stability
        sub_steps = 4
        for _ in range(sub_steps):
            for i, s1 in enumerate(self.shapes):
                # Apply gravity only to falling shapes
                if s1['is_falling']:
                    s1['vel'][1] += self.GRAVITY / sub_steps * self.fall_speed_multiplier
                
                # Apply friction
                s1['vel'][0] *= self.FRICTION
                s1['vel'][1] *= self.FRICTION
                
                # Update position
                s1['pos'][0] += s1['vel'][0] / sub_steps
                s1['pos'][1] += s1['vel'][1] / sub_steps

                # Wall collisions
                if s1['pos'][0] - s1['radius'] < 0:
                    s1['pos'][0] = s1['radius']
                    s1['vel'][0] *= -self.WALL_BOUNCINESS
                elif s1['pos'][0] + s1['radius'] > self.WIDTH:
                    s1['pos'][0] = self.WIDTH - s1['radius']
                    s1['vel'][0] *= -self.WALL_BOUNCINESS
                
                # Floor collision
                if s1['pos'][1] + s1['radius'] > self.HEIGHT:
                    s1['pos'][1] = self.HEIGHT - s1['radius']
                    s1['vel'][1] *= -self.WALL_BOUNCINESS

            # Shape-shape collisions
            for i in range(len(self.shapes)):
                for j in range(i + 1, len(self.shapes)):
                    s1 = self.shapes[i]
                    s2 = self.shapes[j]
                    
                    dx = s2['pos'][0] - s1['pos'][0]
                    dy = s2['pos'][1] - s1['pos'][1]
                    dist_sq = dx*dx + dy*dy
                    min_dist = s1['radius'] + s2['radius']
                    
                    if dist_sq < min_dist * min_dist and dist_sq > 0:
                        dist = math.sqrt(dist_sq)
                        overlap = (min_dist - dist) / 2.0
                        
                        nx = dx / dist
                        ny = dy / dist
                        
                        s1['pos'][0] -= overlap * nx
                        s1['pos'][1] -= overlap * ny
                        s2['pos'][0] += overlap * nx
                        s2['pos'][1] += overlap * ny

                        # Simple collision response
                        s1_mass = s1['radius']**2
                        s2_mass = s2['radius']**2
                        
                        # Only apply impulse if one is falling, or both are moving towards each other
                        relative_vel_dot_norm = (s2['vel'][0] - s1['vel'][0]) * nx + (s2['vel'][1] - s1['vel'][1]) * ny
                        if s1['is_falling'] or s2['is_falling'] or relative_vel_dot_norm < 0:
                            impulse = (2 * relative_vel_dot_norm) / (s1_mass + s2_mass)
                            
                            s1['vel'][0] += impulse * s2_mass * nx * self.SHAPE_BOUNCINESS
                            s1['vel'][1] += impulse * s2_mass * ny * self.SHAPE_BOUNCINESS
                            s2['vel'][0] -= impulse * s1_mass * nx * self.SHAPE_BOUNCINESS
                            s2['vel'][1] -= impulse * s1_mass * ny * self.SHAPE_BOUNCINESS


        # Settling logic
        if self.falling_shape_idx is not None:
            shape = self.shapes[self.falling_shape_idx]
            speed = math.hypot(shape['vel'][0], shape['vel'][1])
            
            if speed < self.SETTLE_VEL_THRESHOLD:
                shape['settle_timer'] += 1
            else:
                shape['settle_timer'] = 0

            if shape['settle_timer'] >= self.SETTLE_FRAMES_REQ:
                shape['is_falling'] = False
                # sfx: shape_land.wav
                if shape['pos'][1] - shape['radius'] < self.DANGER_ZONE_Y:
                    self.game_over = True
                    # sfx: game_over.wav
                self.falling_shape_idx = None


    def _handle_merges(self):
        merged_this_frame = True
        total_merge_reward = 0
        
        while merged_this_frame:
            merged_this_frame = False
            to_remove = set()
            to_add = []
            
            # Find groups of touching, same-level shapes
            adj = defaultdict(list)
            for i in range(len(self.shapes)):
                for j in range(i + 1, len(self.shapes)):
                    s1, s2 = self.shapes[i], self.shapes[j]
                    if s1['level'] == s2['level'] and s1['level'] < self.MAX_LEVEL:
                        dist = math.hypot(s1['pos'][0] - s2['pos'][0], s1['pos'][1] - s2['pos'][1])
                        if dist < s1['radius'] + s2['radius'] + 2: # +2 pixel tolerance
                            adj[i].append(j)
                            adj[j].append(i)

            visited = set()
            for i in range(len(self.shapes)):
                if i not in visited:
                    group = []
                    q = [i]
                    visited.add(i)
                    head = 0
                    while head < len(q):
                        u = q[head]
                        head += 1
                        group.append(u)
                        for v in adj[u]:
                            if v not in visited:
                                visited.add(v)
                                q.append(v)
                    
                    if len(group) >= 2:
                        merged_this_frame = True
                        
                        # Calculate properties of new shape
                        avg_pos = [0, 0]
                        total_area = 0
                        level = self.shapes[group[0]]['level']
                        
                        for idx in group:
                            shape = self.shapes[idx]
                            area = math.pi * shape['radius']**2
                            avg_pos[0] += shape['pos'][0] * area
                            avg_pos[1] += shape['pos'][1] * area
                            total_area += area
                            to_remove.add(idx)

                        avg_pos[0] /= total_area
                        avg_pos[1] /= total_area

                        new_level = level + 1
                        new_radius, new_color = self._get_shape_props(new_level)
                        
                        new_shape = {
                            'pos': avg_pos, 'vel': [0, 0], 'level': new_level,
                            'radius': new_radius, 'color': new_color,
                            'is_falling': True, 'settle_timer': 0, # Let it resettle
                        }
                        to_add.append(new_shape)

                        # Scoring and reward
                        merge_score = len(group) * level
                        self.score += merge_score
                        total_merge_reward += 0.1 * len(group)

                        # Particles
                        self._create_particles(avg_pos, new_color, 20 + 5 * level)
                        # sfx: merge.wav

            if merged_this_frame:
                # Rebuild shapes list
                self.shapes = [s for i, s in enumerate(self.shapes) if i not in to_remove]
                self.shapes.extend(to_add)
                # After a merge, no shape is controlled by the player until things settle
                self.falling_shape_idx = None
                # Make all newly created shapes falling
                for i in range(len(self.shapes) - len(to_add), len(self.shapes)):
                    self.shapes[i]['is_falling'] = True

        return total_merge_reward

    def _check_level_up(self):
        if self.score >= self.level_up_score_target and self.level < self.MAX_LEVEL:
            self.level += 1
            self._update_level_vars()
            self._create_particles([self.WIDTH/2, self.HEIGHT/2], (255, 255, 100), 100)
            # sfx: level_up.wav
            return 5.0 # Level up reward
        return 0.0

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': self.np_random.integers(15, 30),
                'color': color,
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_shapes()
        self._render_particles()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        
        # Danger Zone
        danger_rect = pygame.Surface((self.WIDTH, self.DANGER_ZONE_Y), pygame.SRCALPHA)
        danger_rect.fill(self.COLOR_DANGER)
        self.screen.blit(danger_rect, (0,0))
        pygame.draw.line(self.screen, (255, 80, 80), (0, self.DANGER_ZONE_Y), (self.WIDTH, self.DANGER_ZONE_Y), 2)


    def _render_shapes(self):
        # Draw drop line for falling shape
        if self.falling_shape_idx is not None:
            shape = self.shapes[self.falling_shape_idx]
            x = int(shape['pos'][0])
            pygame.draw.line(self.screen, (255, 255, 255, 50), (x, int(shape['pos'][1])), (x, self.HEIGHT), 1)

        for i, shape in enumerate(self.shapes):
            pos = (int(shape['pos'][0]), int(shape['pos'][1]))
            radius = int(shape['radius'])
            
            # Pulsing outline for falling shape
            if i == self.falling_shape_idx:
                pulse_rad = radius + 2 + int(math.sin(self.steps * 0.2) * 2)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], pulse_rad, (255, 255, 255, 80))
            
            # Main shape body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, shape['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, shape['color'])
            
            # Level number text
            if radius > 12:
                font_size = max(12, int(radius * 0.8))
                level_font = pygame.font.Font(None, font_size)
                text = level_font.render(str(shape['level']), True, (255,255,255) if sum(shape['color']) < 384 else (0,0,0))
                text_rect = text.get_rect(center=pos)
                self.screen.blit(text, text_rect)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30.0))
            color = (*p['color'], alpha)
            rect = pygame.Rect(int(p['pos'][0]), int(p['pos'][1]), 2, 2)
            # Use a small surface for alpha blending
            particle_surf = pygame.Surface((2, 2), pygame.SRCALPHA)
            particle_surf.fill(color)
            self.screen.blit(particle_surf, rect.topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 10))
        
        # Level
        level_text = self.font_main.render(f"Level: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (15, 10))

        # Next Shape Preview
        preview_box_size = 80
        preview_x = self.WIDTH // 2 - preview_box_size // 2
        preview_y = self.HEIGHT - preview_box_size - 10
        
        preview_rect = pygame.Rect(preview_x, preview_y, preview_box_size, preview_box_size)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, preview_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.COLOR_PREVIEW_OUTLINE, preview_rect, 2, border_radius=10)

        next_text = self.font_small.render("Next", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (preview_rect.centerx - next_text.get_width() // 2, preview_rect.top + 5))
        
        if self.next_shape_level is not None:
            radius, color = self._get_shape_props(self.next_shape_level)
            pos = (preview_rect.centerx, preview_rect.centery + 10)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius), color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(radius), color)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text = "WINNER!" if self.level >= self.WIN_LEVEL else "GAME OVER"
        color = (100, 255, 100) if self.level >= self.WIN_LEVEL else (255, 100, 100)
        
        game_over_text = self.font_game_over.render(text, True, color)
        text_rect = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "level": self.level,
            "steps": self.steps,
            "shape_count": len(self.shapes),
        }

    def _generate_color_palette(self, n_colors):
        """Generates a visually distinct, vibrant color palette."""
        palette = []
        for i in range(n_colors):
            hue = i / n_colors
            # Use sine waves for smooth, non-linear color transitions
            r = int((math.sin(hue * 2 * math.pi + 0) * 0.5 + 0.5) * 205 + 50)
            g = int((math.sin(hue * 2 * math.pi + 2 * math.pi / 3) * 0.5 + 0.5) * 205 + 50)
            b = int((math.sin(hue * 2 * math.pi + 4 * math.pi / 3) * 0.5 + 0.5) * 205 + 50)
            palette.append((r, g, b))
        return palette
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # The main script sets SDL_VIDEODRIVER to "dummy", so this will not create a window.
    # To play manually, comment out the os.environ.setdefault line at the top of the file.
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Shape Merger")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Human Controls ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2

        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.metadata['render_fps'])

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()