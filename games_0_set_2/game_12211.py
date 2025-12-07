import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:31:44.052428
# Source Brief: brief_02211.md
# Brief Index: 2211
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Fractal Time Manipulator

    A Gymnasium environment where the agent must grow a fractal structure to match a
    target pattern. The agent can place and activate temporal portals to create
    non-local branches, manipulating the growth process. The environment is designed
    with a strong emphasis on visual quality and engaging gameplay.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Grow a fractal structure to match a target pattern. Place and activate temporal portals "
        "to manipulate the growth process and create non-local branches."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place a portal and "
        "shift to activate/deactivate the nearest one."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    MAX_BRANCHES = 400
    CURSOR_SPEED = 8
    PORTAL_RADIUS = 15
    PORTAL_INTERACTION_RADIUS = 20
    MAX_PORTALS = 5
    FONT_SIZE_UI = 24
    FONT_SIZE_TITLE = 16

    # --- COLORS (VIBRANT & CONTRASTING) ---
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER_CURSOR = (255, 255, 0)
    COLOR_TARGET_FRACTAL = (0, 100, 50) # Muted green for background
    COLOR_GROWING_FRACTAL = (100, 200, 255) # Bright cyan/blue
    COLOR_PORTAL_INACTIVE = (100, 100, 120)
    COLOR_PORTAL_ACTIVE = (255, 150, 0) # Bright orange
    COLOR_PORTAL_GLOW = (255, 150, 0)
    COLOR_TIME_RIPPLE = (150, 50, 255) # Purple
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_SCORE_POSITIVE = (100, 255, 100)
    COLOR_SCORE_NEGATIVE = (255, 100, 100)
    
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
        self.ui_font = pygame.font.Font(None, self.FONT_SIZE_UI)
        self.title_font = pygame.font.Font(None, self.FONT_SIZE_TITLE)

        self.steps = 0
        self.score = 0
        self.level = 1
        
        # Game state variables, initialized in reset
        self.cursor_pos = None
        self.portals = None
        self.portals_remaining = None
        self.branches = None
        self.target_lines = None
        self.unmatched_target_pixels = None
        self.matched_pixels = None
        self.particles = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.game_over = None
        self.total_branch_segments = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.total_branch_segments = 1

        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.portals = []
        self.portals_remaining = self.MAX_PORTALS
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self._generate_target_fractal()
        
        # Initial growing fractal branch
        start_pos = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20)
        self.branches = [{
            "start": np.array(start_pos, dtype=np.float32),
            "angle": -math.pi / 2,
            "length": 0,
            "max_length": 60,
            "depth": 0,
            "thickness": 4,
            "is_done": False,
            "teleported": False
        }]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle Input ---
        reward += self._handle_input(movement, space_held, shift_held)

        # --- Update Game State ---
        new_pixels, portal_reward = self._update_fractal_growth()
        reward += portal_reward
        self._update_particles()
        
        # --- Calculate Pixel-based Reward ---
        for px, py in new_pixels:
            if (px, py) in self.unmatched_target_pixels:
                reward += 0.1
                self.score += 0.1
                self.unmatched_target_pixels.remove((px, py))
                self.matched_pixels.add((px, py))
            elif (px, py) not in self.matched_pixels:
                reward -= 0.01
                self.score -= 0.01
        
        self.steps += 1

        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if not self.unmatched_target_pixels:
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
            # Win effect
            self._spawn_particles(self.cursor_pos, 100, (100, 255, 100), 5)
        elif self.steps >= self.MAX_STEPS or self.total_branch_segments >= self.MAX_BRANCHES:
            reward -= 50
            self.score -= 50
            terminated = True
            self.game_over = True
            # Lose effect
            self._spawn_particles(self.cursor_pos, 50, (255, 100, 100), 3)

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

        # Place portal (on press)
        if space_held and not self.prev_space_held:
            if self.portals_remaining > 0:
                can_place = True
                for p in self.portals:
                    if np.linalg.norm(self.cursor_pos - p['pos']) < self.PORTAL_RADIUS * 2:
                        can_place = False
                        break
                if can_place:
                    self.portals.append({'pos': self.cursor_pos.copy(), 'active': False, 'id': len(self.portals)})
                    self.portals_remaining -= 1
                    reward -= 0.5 # Cost for using a portal
                    # sfx: portal_place.wav
                    self._spawn_particles(self.cursor_pos, 20, self.COLOR_PORTAL_INACTIVE, 2)

        # Activate/deactivate portal (on press)
        if shift_held and not self.prev_shift_held:
            if self.portals:
                distances = [np.linalg.norm(self.cursor_pos - p['pos']) for p in self.portals]
                closest_portal_idx = np.argmin(distances)
                if distances[closest_portal_idx] < self.PORTAL_INTERACTION_RADIUS * 2:
                    self.portals[closest_portal_idx]['active'] = not self.portals[closest_portal_idx]['active']
                    # sfx: portal_activate.wav / portal_deactivate.wav
                    color = self.COLOR_PORTAL_ACTIVE if self.portals[closest_portal_idx]['active'] else self.COLOR_PORTAL_INACTIVE
                    self._spawn_particles(self.portals[closest_portal_idx]['pos'], 30, color, 3)

        return reward
        
    def _update_fractal_growth(self):
        newly_grown_pixels = set()
        portal_reward = 0
        
        new_branches = []
        active_portals = [p for p in self.portals if p['active']]

        for branch in self.branches:
            if branch['is_done']:
                continue

            growth_speed = 1.0
            p1 = branch['start']
            old_end = p1 + branch['length'] * np.array([math.cos(branch['angle']), math.sin(branch['angle'])])

            branch['length'] += growth_speed
            
            p2 = p1 + branch['length'] * np.array([math.cos(branch['angle']), math.sin(branch['angle'])])
            
            # Add newly grown pixels for reward calculation
            for pixel in self._get_line_pixels(old_end, p2):
                newly_grown_pixels.add(pixel)

            # Portal interaction
            if not branch['teleported'] and len(active_portals) >= 2:
                for portal in active_portals:
                    if np.linalg.norm(p2 - portal['pos']) < self.PORTAL_INTERACTION_RADIUS:
                        branch['teleported'] = True # Prevent multiple teleports from one branch
                        
                        # Find a destination portal (not the one we entered)
                        dest_portals = [p for p in active_portals if p['id'] != portal['id']]
                        dest_portal = self.np_random.choice(dest_portals)
                        
                        # Create a new branch from the destination
                        new_angle = branch['angle'] + self.np_random.uniform(-math.pi/8, math.pi/8)
                        new_branch = {
                            "start": dest_portal['pos'].copy(),
                            "angle": new_angle,
                            "length": 0,
                            "max_length": branch['max_length'] * 0.8,
                            "depth": branch['depth'] + 1,
                            "thickness": max(1, branch['thickness'] - 1),
                            "is_done": False,
                            "teleported": False
                        }
                        if new_branch['max_length'] > 5:
                            new_branches.append(new_branch)
                            self.total_branch_segments += 1
                            # sfx: teleport.wav
                            self._spawn_particles(portal['pos'], 15, self.COLOR_TIME_RIPPLE, 2)
                            self._spawn_particles(dest_portal['pos'], 15, self.COLOR_TIME_RIPPLE, 2)
                            portal_reward += 2.0 # Reward for using the portal system
                        break

            # Branch completion and spawning children
            if branch['length'] >= branch['max_length']:
                branch['is_done'] = True
                # sfx: branch_complete.wav
                self._spawn_particles(p2, 10, self.COLOR_GROWING_FRACTAL, 1)

                if branch['depth'] < self.level + 2: # Control complexity
                    num_children = 2
                    for i in range(num_children):
                        angle_offset = self.np_random.uniform(math.pi / 8, math.pi / 4) * (1 if i == 0 else -1)
                        child_branch = {
                            "start": p2.copy(),
                            "angle": branch['angle'] + angle_offset,
                            "length": 0,
                            "max_length": branch['max_length'] * self.np_random.uniform(0.6, 0.8),
                            "depth": branch['depth'] + 1,
                            "thickness": max(1, branch['thickness'] - 1),
                            "is_done": False,
                            "teleported": False
                        }
                        if child_branch['max_length'] > 5: # Don't create tiny branches
                           new_branches.append(child_branch)
                           self.total_branch_segments += 1

        self.branches.extend(new_branches)
        return newly_grown_pixels, portal_reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1

    def _generate_target_fractal(self):
        self.target_lines = []
        self.unmatched_target_pixels = set()
        self.matched_pixels = set()
        
        # Difficulty scales with level
        max_depth = min(5, 2 + self.level // 2)
        initial_length = 60 + (self.level % 2) * 10
        
        start_pos = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20)
        
        # Use a separate random number generator for deterministic level generation
        level_rng = random.Random(self.level)
        
        def _create_branches(start, angle, length, depth):
            if depth > max_depth or length < 10:
                return
            
            end = (start[0] + length * math.cos(angle), start[1] + length * math.sin(angle))
            self.target_lines.append((start, end))
            
            for pixel in self._get_line_pixels(np.array(start), np.array(end)):
                self.unmatched_target_pixels.add(pixel)

            num_children = 2 if depth < 3 else level_rng.choice([1, 2])
            for i in range(num_children):
                angle_dev = level_rng.uniform(math.pi / 9, math.pi / 4)
                new_angle = angle + angle_dev * (1 if i == 0 else -1)
                new_length = length * level_rng.uniform(0.65, 0.85)
                _create_branches(end, new_angle, new_length, depth + 1)

        _create_branches(start_pos, -math.pi / 2, initial_length, 0)
    
    def _get_line_pixels(self, p1, p2):
        pixels = []
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        dx, dy = abs(x2 - x1), -abs(y2 - y1)
        sx, sy = 1 if x1 < x2 else -1, 1 if y1 < y2 else -1
        err = dx + dy
        while True:
            if 0 <= x1 < self.SCREEN_WIDTH and 0 <= y1 < self.SCREEN_HEIGHT:
                pixels.append((x1, y1))
            if x1 == x2 and y1 == y2: break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy
        return pixels

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background stars
        for i in range(100):
            rng = random.Random(i)
            color_val = rng.randint(40, 80)
            pygame.gfxdraw.pixel(self.screen, 
                                 rng.randint(0, self.SCREEN_WIDTH), 
                                 rng.randint(0, self.SCREEN_HEIGHT), 
                                 (color_val, color_val, color_val))

        # Render target fractal
        for p1, p2 in self.target_lines:
            pygame.draw.line(self.screen, self.COLOR_TARGET_FRACTAL, p1, p2, 1)

        # Render growing fractal
        for branch in self.branches:
            start_pos = branch['start']
            end_pos = start_pos + branch['length'] * np.array([math.cos(branch['angle']), math.sin(branch['angle'])])
            if branch['length'] > 0:
                pygame.draw.line(self.screen, self.COLOR_GROWING_FRACTAL, 
                                 start_pos.astype(int), end_pos.astype(int), 
                                 max(1, int(branch['thickness'])))

        # Render portals
        for portal in self.portals:
            pos = portal['pos'].astype(int)
            if portal['active']:
                # Time ripple effect
                ripple_rad = (self.PORTAL_RADIUS * 1.5 + 5 * math.sin(self.steps * 0.2))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(ripple_rad), (*self.COLOR_TIME_RIPPLE, 100))
                # Glow effect
                self._draw_glowing_circle(self.screen, self.COLOR_PORTAL_GLOW, pos, self.PORTAL_RADIUS, 15)
                # Main circle
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PORTAL_RADIUS, self.COLOR_PORTAL_ACTIVE)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PORTAL_RADIUS, self.COLOR_PORTAL_ACTIVE)
            else:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PORTAL_RADIUS, self.COLOR_PORTAL_INACTIVE)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PORTAL_RADIUS, self.COLOR_PORTAL_INACTIVE)
        
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / p['max_lifespan']))))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

        # Render cursor
        c_pos = self.cursor_pos.astype(int)
        self._draw_glowing_circle(self.screen, self.COLOR_PLAYER_CURSOR, c_pos, 8, 10)
        pygame.draw.line(self.screen, self.COLOR_PLAYER_CURSOR, (c_pos[0] - 5, c_pos[1]), (c_pos[0] + 5, c_pos[1]), 1)
        pygame.draw.line(self.screen, self.COLOR_PLAYER_CURSOR, (c_pos[0], c_pos[1] - 5), (c_pos[0], c_pos[1] + 5), 1)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score:.1f}"
        score_color = self.COLOR_SCORE_POSITIVE if self.score >= 0 else self.COLOR_SCORE_NEGATIVE
        self._draw_text(score_text, (10, 10), score_color, self.ui_font)

        # Portals remaining
        portal_text = f"PORTALS: {self.portals_remaining}"
        self._draw_text(portal_text, (self.SCREEN_WIDTH - 150, 10), self.COLOR_UI_TEXT, self.ui_font)
        
        # Level
        level_text = f"LEVEL: {self.level}"
        self._draw_text(level_text, (self.SCREEN_WIDTH / 2 - 40, 10), self.COLOR_UI_TEXT, self.ui_font)

        # Target preview
        preview_rect = pygame.Rect(self.SCREEN_WIDTH - 105, self.SCREEN_HEIGHT - 105, 100, 100)
        pygame.draw.rect(self.screen, (50, 50, 70), preview_rect, 1)
        self._draw_text("TARGET", (preview_rect.x + 28, preview_rect.y - 15), self.COLOR_UI_TEXT, self.title_font)
        
        if self.target_lines:
            min_x = min(p[0] for line in self.target_lines for p in line)
            max_x = max(p[0] for line in self.target_lines for p in line)
            min_y = min(p[1] for line in self.target_lines for p in line)
            max_y = max(p[1] for line in self.target_lines for p in line)
            
            target_w = max_x - min_x
            target_h = max_y - min_y
            if target_w > 0 and target_h > 0:
                scale = min(90 / target_w, 90 / target_h)
                for p1, p2 in self.target_lines:
                    sp1 = ((p1[0] - min_x) * scale + preview_rect.x + 5, (p1[1] - min_y) * scale + preview_rect.y + 5)
                    sp2 = ((p2[0] - min_x) * scale + preview_rect.x + 5, (p2[1] - min_y) * scale + preview_rect.y + 5)
                    pygame.draw.line(self.screen, self.COLOR_TARGET_FRACTAL, sp1, sp2, 1)

    def _draw_text(self, text, pos, color, font):
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _draw_glowing_circle(self, surface, color, center, radius, max_glow):
        for i in range(max_glow, 0, -1):
            alpha = int(100 * (1 - (i / max_glow)))
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius + i, glow_color)

    def _spawn_particles(self, pos, count, color, speed_max):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': pos.copy().astype(np.float32),
                'vel': vel,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "portals_remaining": self.portals_remaining,
            "match_progress": len(self.matched_pixels) / (len(self.unmatched_target_pixels) + len(self.matched_pixels) + 1e-6)
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # The `validate_implementation` method is removed as it's for internal testing
    # and not part of the standard Gym API. The __main__ block provides a
    # simple way to run and play the game.
    
    # Set a non-dummy driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 

    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    pygame.display.set_caption("Fractal Time Manipulator")
    human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    while running:
        action = [0, 0, 0] # no-op
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Environment Reset ---")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Final Info: {info}")
            obs, info = env.reset()
            total_reward = 0
        
        # Render to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for playability

    env.close()