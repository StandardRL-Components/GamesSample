
# Generated: 2025-08-27T12:46:50.065247
# Source Brief: brief_00158.md
# Brief Index: 158

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys move the drawing cursor. "
        "Spacebar places a line. Shift clears the last line."
    )
    game_description = (
        "Draw lines to guide a physics-based rider across procedurally generated "
        "terrain. Reach the finish line before time runs out, collecting checkpoints "
        "for extra time. Don't let the rider fall!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG_TOP = (15, 25, 40)
    COLOR_BG_BOTTOM = (30, 50, 70)
    COLOR_RIDER = (255, 80, 80)
    COLOR_RIDER_GLOW = (255, 120, 120, 50)
    COLOR_LINE = (255, 255, 255)
    COLOR_GHOST_LINE = (200, 200, 200, 100)
    COLOR_TERRAIN = (100, 120, 140)
    COLOR_CHECKPOINT = (255, 220, 0)
    COLOR_FINISH = (80, 255, 80)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_PARTICLE = (255, 100, 100)

    # Physics & Game Rules
    GRAVITY = pygame.math.Vector2(0, 0.4)
    RIDER_RADIUS = 10
    MAX_LINES = 15
    INITIAL_TIME = 30.0
    CHECKPOINT_BONUS = 5.0
    FINISH_X = 15000
    MAX_STEPS = 2000
    CURSOR_SPEED = 8
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 40)
            self.font_small = pygame.font.Font(None, 24)

        self.render_mode = render_mode
        self.game_over = True
        
        # This will be properly initialized in reset()
        self.rider_pos = pygame.math.Vector2(0, 0)
        self.rider_vel = pygame.math.Vector2(0, 0)
        self.drawn_lines = []
        self.terrain_points = deque()
        self.checkpoints = []
        self.particles = []
        self.camera_x = 0.0
        self.cursor_start = pygame.math.Vector2(0, 0)
        self.cursor_end = pygame.math.Vector2(0, 0)
        self.steps = 0
        self.score = 0
        self.timer = 0.0
        self.checkpoints_passed = 0
        self.last_rider_x = 0.0
        self.action_cooldowns = {'draw': 0, 'clear': 0}

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.INITIAL_TIME
        self.checkpoints_passed = 0
        
        self.rider_pos = pygame.math.Vector2(100, 150)
        self.rider_vel = pygame.math.Vector2(3, 0)
        self.last_rider_x = self.rider_pos.x
        
        self.drawn_lines = []
        self.particles = []
        self.camera_x = 0.0
        
        self.cursor_start = self.rider_pos + (50, 0)
        self.cursor_end = self.rider_pos + (150, 0)

        self.action_cooldowns = {'draw': 0, 'clear': 0}
        
        self._generate_initial_world()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_pressed, shift_pressed)
        self._update_physics()
        self._update_game_state()
        
        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self._check_termination()
        self.game_over = terminated
        
        if terminated:
            if self.rider_pos.x >= self.FINISH_X:
                reward += 100.0 # Victory bonus
            else:
                reward -= 100.0 # Crash/Timeout penalty
                if self.rider_pos.y > self.SCREEN_HEIGHT + 20 or self.rider_pos.y < -20:
                    self._create_particles(self.rider_pos, 30) # Crash explosion
            self.score += reward

        self.steps += 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Update cooldowns
        for key in self.action_cooldowns:
            self.action_cooldowns[key] = max(0, self.action_cooldowns[key] - 1)

        # Move cursor
        cursor_offset = self.cursor_end - self.cursor_start
        if movement == 1: cursor_offset.y -= self.CURSOR_SPEED  # Up
        if movement == 2: cursor_offset.y += self.CURSOR_SPEED  # Down
        if movement == 3: cursor_offset.x -= self.CURSOR_SPEED  # Left
        if movement == 4: cursor_offset.x += self.CURSOR_SPEED  # Right
        
        # Clamp cursor length
        if cursor_offset.length() > 200:
            cursor_offset.scale_to_length(200)
        if cursor_offset.length() < 20:
            cursor_offset.scale_to_length(20)
        
        self.cursor_end = self.cursor_start + cursor_offset

        # Draw line
        if space_pressed and self.action_cooldowns['draw'] == 0:
            self.drawn_lines.append((self.cursor_start.copy(), self.cursor_end.copy()))
            if len(self.drawn_lines) > self.MAX_LINES:
                self.drawn_lines.pop(0)
            self.action_cooldowns['draw'] = 10 # 1/3 second cooldown
            # sfx: line_draw.wav

        # Clear line
        if shift_pressed and self.action_cooldowns['clear'] == 0 and self.drawn_lines:
            self.drawn_lines.pop()
            self.action_cooldowns['clear'] = 15 # 1/2 second cooldown
            # sfx: line_clear.wav

    def _update_physics(self):
        # Apply gravity
        self.rider_vel += self.GRAVITY
        
        # Move rider
        self.rider_pos += self.rider_vel

        # Collision detection and response
        all_lines = self.drawn_lines + list(zip(self.terrain_points, list(self.terrain_points)[1:]))
        
        collided = False
        for p1, p2 in all_lines:
            p1_vec, p2_vec = pygame.math.Vector2(p1), pygame.math.Vector2(p2)
            line_vec = p2_vec - p1_vec
            if line_vec.length() == 0: continue

            point_vec = self.rider_pos - p1_vec
            t = point_vec.dot(line_vec) / line_vec.dot(line_vec)
            t = max(0, min(1, t))
            
            closest_point = p1_vec + t * line_vec
            dist_vec = self.rider_pos - closest_point
            
            if dist_vec.length() < self.RIDER_RADIUS:
                collided = True
                # sfx: grind.wav
                
                # Resolve penetration
                overlap = self.RIDER_RADIUS - dist_vec.length()
                self.rider_pos += dist_vec.normalize() * overlap
                
                # Collision response (reflection and friction)
                normal = dist_vec.normalize()
                reflect_vel = self.rider_vel.reflect(normal)
                
                # Friction
                friction = 0.95
                self.rider_vel = reflect_vel * friction

                # Add grinding particles
                if self.np_random.random() < 0.5:
                    self._create_particles(closest_point, 1, speed_mult=0.5)
                break
    
    def _update_game_state(self):
        # Update timer
        self.timer = max(0, self.timer - 1.0 / self.FPS)
        
        # Update camera
        target_cam_x = self.rider_pos.x - self.SCREEN_WIDTH / 3
        self.camera_x += (target_cam_x - self.camera_x) * 0.1

        # Update cursor position
        self.cursor_start = self.rider_pos + (80, 0)
        self.cursor_end += self.rider_pos - (self.cursor_start - (80,0))

        # Generate new terrain
        last_terrain_x = self.terrain_points[-1][0]
        if last_terrain_x < self.camera_x + self.SCREEN_WIDTH + 200:
            self._generate_terrain_segment()

        # Remove old terrain/lines/checkpoints
        self.terrain_points = deque(p for p in self.terrain_points if p[0] > self.camera_x - 100)
        self.drawn_lines = [l for l in self.drawn_lines if l[1].x > self.camera_x - 100]
        
        # Check for passing checkpoints
        remaining_checkpoints = []
        for cp_pos, cp_radius in self.checkpoints:
            if self.rider_pos.distance_to(cp_pos) < self.RIDER_RADIUS + cp_radius:
                self.checkpoints_passed += 1
                self.timer += self.CHECKPOINT_BONUS
                self.score += 1.0 # Checkpoint reward
                self._create_particles(cp_pos, 20, color=(255, 220, 0))
                # sfx: checkpoint.wav
            elif cp_pos.x > self.camera_x - 100:
                remaining_checkpoints.append((cp_pos, cp_radius))
        self.checkpoints = remaining_checkpoints

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _calculate_reward(self):
        # Reward for forward progress
        progress = self.rider_pos.x - self.last_rider_x
        reward = progress * 0.1
        self.last_rider_x = self.rider_pos.x
        
        # Small penalty for time passing
        reward -= 0.01
        
        return reward

    def _check_termination(self):
        # Rider fell off screen
        if self.rider_pos.y > self.SCREEN_HEIGHT + self.RIDER_RADIUS * 2 or self.rider_pos.y < -self.RIDER_RADIUS * 5:
            return True
        # Reached finish line
        if self.rider_pos.x >= self.FINISH_X:
            return True
        # Time ran out
        if self.timer <= 0:
            return True
        # Max steps reached
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self._draw_gradient_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "checkpoints": self.checkpoints_passed,
            "distance": self.rider_pos.x
        }

    def _render_game(self):
        # Render terrain
        terrain_screen_points = [(p[0] - self.camera_x, p[1]) for p in self.terrain_points]
        if len(terrain_screen_points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_TERRAIN, False, terrain_screen_points, 2)

        # Render checkpoints
        for cp_pos, cp_radius in self.checkpoints:
            sx, sy = int(cp_pos.x - self.camera_x), int(cp_pos.y)
            if -cp_radius < sx < self.SCREEN_WIDTH + cp_radius:
                pygame.gfxdraw.aacircle(self.screen, sx, sy, int(cp_radius), self.COLOR_CHECKPOINT)
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, int(cp_radius), self.COLOR_CHECKPOINT)

        # Render finish line
        finish_sx = self.FINISH_X - self.camera_x
        if 0 < finish_sx < self.SCREEN_WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_sx, 0), (finish_sx, self.SCREEN_HEIGHT), 5)

        # Render drawn lines
        for p1, p2 in self.drawn_lines:
            pygame.draw.aaline(self.screen, self.COLOR_LINE, 
                               (p1.x - self.camera_x, p1.y), 
                               (p2.x - self.camera_x, p2.y), 2)

        # Render ghost line
        if not self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.aaline(s, self.COLOR_GHOST_LINE,
                               (self.cursor_start.x - self.camera_x, self.cursor_start.y),
                               (self.cursor_end.x - self.camera_x, self.cursor_end.y), 2)
            self.screen.blit(s, (0, 0))

        # Render particles
        for p in self.particles:
            pos = (int(p['pos'].x - self.camera_x), int(p['pos'].y))
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (pos[0] - p['size'], pos[1] - p['size']), special_flags=pygame.BLEND_RGBA_ADD)

        # Render rider
        rider_sx = int(self.rider_pos.x - self.camera_x)
        rider_sy = int(self.rider_pos.y)
        
        # Glow effect
        glow_surf = pygame.Surface((self.RIDER_RADIUS * 4, self.RIDER_RADIUS * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_RIDER_GLOW, (self.RIDER_RADIUS * 2, self.RIDER_RADIUS * 2), self.RIDER_RADIUS * 1.5)
        self.screen.blit(glow_surf, (rider_sx - self.RIDER_RADIUS * 2, rider_sy - self.RIDER_RADIUS * 2), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.gfxdraw.aacircle(self.screen, rider_sx, rider_sy, self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.filled_circle(self.screen, rider_sx, rider_sy, self.RIDER_RADIUS, self.COLOR_RIDER)

    def _render_ui(self):
        # Timer
        timer_text = f"TIME: {self.timer:.1f}"
        text_surf = self.font_large.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (20, 10))

        # Speed
        speed = self.rider_vel.length() * 10 
        speed_text = f"SPD: {speed:.0f}"
        text_surf = self.font_large.render(speed_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 20, 10))

        # Checkpoints
        cp_text = f"CP: {self.checkpoints_passed}"
        text_surf = self.font_small.render(cp_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (20, 50))

    def _draw_gradient_background(self):
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
            
    def _generate_initial_world(self):
        self.terrain_points = deque()
        self.checkpoints = []
        
        current_x = -200
        current_y = self.SCREEN_HEIGHT / 2 + 100
        
        # Flat starting area
        for _ in range(5):
            self.terrain_points.append((current_x, current_y))
            current_x += 100
        
        while current_x < self.FINISH_X + self.SCREEN_WIDTH:
            self.terrain_points.append((current_x, current_y))
            num_segments = self.np_random.integers(1, 4)
            for _ in range(num_segments):
                difficulty = self.checkpoints_passed // 5
                max_gap = 50 + difficulty * 5
                max_angle = 20 + difficulty * 2
                
                length = self.np_random.uniform(100, 300)
                angle = self.np_random.uniform(-max_angle, max_angle)
                
                current_x += length * math.cos(math.radians(angle))
                current_y += length * math.sin(math.radians(angle))
                current_y = np.clip(current_y, 100, self.SCREEN_HEIGHT - 50)
                self.terrain_points.append((current_x, current_y))
            
            # Add a checkpoint every ~1000 pixels
            if len(self.checkpoints) == 0 or current_x - self.checkpoints[-1][0].x > 1000:
                cp_pos = pygame.math.Vector2(current_x - 150, current_y - 80)
                self.checkpoints.append((cp_pos, 15))

            # Add a gap
            gap_size = self.np_random.uniform(50, max_gap)
            current_x += gap_size
            current_y += self.np_random.uniform(-50, 50)
            current_y = np.clip(current_y, 100, self.SCREEN_HEIGHT - 50)

    def _generate_terrain_segment(self):
        current_x, current_y = self.terrain_points[-1]
        
        difficulty = (self.checkpoints_passed + (current_x // 5000)) // 5
        max_gap = 50 + difficulty * 10
        max_angle = 20 + difficulty * 3
        
        num_segments = self.np_random.integers(1, 4)
        for _ in range(num_segments):
            length = self.np_random.uniform(100, 300)
            angle = self.np_random.uniform(-max_angle, max_angle)
            
            current_x += length * math.cos(math.radians(angle))
            current_y += length * math.sin(math.radians(angle))
            current_y = np.clip(current_y, 100, self.SCREEN_HEIGHT - 50)
            self.terrain_points.append((current_x, current_y))

        # Add a checkpoint
        if current_x - (self.checkpoints[-1][0].x if self.checkpoints else 0) > 1000:
             cp_pos = pygame.math.Vector2(current_x - 150, current_y - 80)
             self.checkpoints.append((cp_pos, 15))

        # Add a gap
        gap_size = self.np_random.uniform(50, max_gap)
        current_x += gap_size
        current_y += self.np_random.uniform(-50, 50)
        current_y = np.clip(current_y, 100, self.SCREEN_HEIGHT - 50)
        self.terrain_points.append((current_x, current_y)) # Add point after gap

    def _create_particles(self, pos, count, speed_mult=1.0, color=None):
        if color is None:
            color = self.COLOR_PARTICLE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) * speed_mult
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            max_life = self.np_random.integers(15, 40)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': max_life,
                'max_life': max_life,
                'size': self.np_random.integers(1, 4),
                'color': color
            })
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Distance: {info['distance']:.0f}")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(GameEnv.FPS)
        
    env.close()