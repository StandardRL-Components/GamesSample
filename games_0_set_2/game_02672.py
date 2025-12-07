
# Generated: 2025-08-27T21:04:53.676797
# Source Brief: brief_02672.md
# Brief Index: 2672

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Hold Space to draw a track segment from the last anchor to your cursor."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a sledder down a procedurally generated track by drawing lines, aiming for speed and style while avoiding crashes."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Colors
        self.COLOR_BG_TOP = (40, 50, 80)
        self.COLOR_BG_BOTTOM = (60, 80, 120)
        self.COLOR_TRACK = (255, 255, 255)
        self.COLOR_PLAYER = (220, 50, 50)
        self.COLOR_PLAYER_GLOW = (255, 100, 100, 50)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_ANCHOR = (150, 150, 255)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_FINISH = (50, 220, 50)

        # Physics and Game constants
        self.GRAVITY = 0.2
        self.FRICTION = 0.995
        self.MAX_STEPS = 2000
        self.CURSOR_SPEED = 8
        self.MIN_DRAW_DIST = 10
        self.RIDER_RADIUS = 8
        self.JUMP_REWARD_SPEED_THRESHOLD = 5.0

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.on_ground = False
        self.current_track_segment = None

        self.cursor_pos = None
        self.line_anchor_pos = None

        self.world_lines = []
        self.player_lines = []
        
        self.particles = []
        self.camera_y = 0
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.crashed = False
        self.finished = False
        self.finish_line_x = 0
        
        self.reset()
        
        self.validate_implementation()

    def _generate_track(self):
        self.world_lines.clear()
        x, y = 50, self.HEIGHT / 2
        start_pos = pygame.Vector2(x, y)
        
        # Starting platform
        self.world_lines.append((pygame.Vector2(x - 50, y), start_pos.copy()))

        num_segments = 20
        for i in range(num_segments):
            px, py = x, y
            x += self.np_random.uniform(80, 150)
            y += self.np_random.uniform(-50, 100)
            # Clamp y to prevent extreme tracks
            y = np.clip(y, 50, self.HEIGHT - 50)
            self.world_lines.append((pygame.Vector2(px, py), pygame.Vector2(x, y)))
        
        self.finish_line_x = x
        self.world_lines.append((pygame.Vector2(x, y), pygame.Vector2(x + 100, y)))
        
        return start_pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        start_pos = self._generate_track()

        self.player_pos = start_pos.copy() + pygame.Vector2(10, -self.RIDER_RADIUS*2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = 0
        self.on_ground = False
        self.current_track_segment = None

        self.cursor_pos = start_pos.copy() + pygame.Vector2(50, 0)
        self.line_anchor_pos = start_pos.copy()
        
        self.player_lines.clear()
        self.particles.clear()
        
        self.camera_y = self.player_pos.y - self.HEIGHT / 2
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.crashed = False
        self.finished = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        
        self._handle_input(movement, space_pressed)
        
        last_x_pos = self.player_pos.x
        was_on_ground = self.on_ground
        
        self._update_player_physics()
        self._update_particles()
        self._update_camera()

        reward = self._calculate_reward(last_x_pos, was_on_ground)
        self.score += reward
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            self.game_over = True
            if self.crashed:
                self.score -= 50
                reward -= 50
                self._create_crash_explosion()
                # sfx_crash
            elif self.finished:
                self.score += 100
                reward += 100
                # sfx_win
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.finish_line_x + 200)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)
        
        if space_pressed:
            dist_sq = self.line_anchor_pos.distance_squared_to(self.cursor_pos)
            if dist_sq > self.MIN_DRAW_DIST ** 2:
                self.player_lines.append((self.line_anchor_pos.copy(), self.cursor_pos.copy()))
                self.line_anchor_pos = self.cursor_pos.copy()
                # sfx_draw_line
                if len(self.player_lines) > 50: # Memory leak prevention
                    self.player_lines.pop(0)

    def _update_player_physics(self):
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY
            self.player_pos += self.player_vel
            self.player_angle = self.player_vel.angle_to(pygame.Vector2(1, 0))
        else:
            self.player_vel *= self.FRICTION

        self.on_ground = False
        
        all_lines = self.world_lines + self.player_lines
        
        best_line = None
        min_dist_sq = float('inf')
        
        for p1, p2 in all_lines:
            # Broad phase check
            if not (min(p1.x, p2.x) - self.RIDER_RADIUS < self.player_pos.x < max(p1.x, p2.x) + self.RIDER_RADIUS):
                continue

            # Closest point calculation
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
            
            p_to_p1 = self.player_pos - p1
            t = p_to_p1.dot(line_vec) / line_vec.length_squared()
            t = np.clip(t, 0, 1)
            
            closest_point = p1 + t * line_vec
            dist_sq = self.player_pos.distance_squared_to(closest_point)

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_line = (p1, p2, closest_point, line_vec)

        if best_line and min_dist_sq < self.RIDER_RADIUS**2:
            p1, p2, closest_point, line_vec = best_line
            self.on_ground = True
            self.current_track_segment = (p1, p2)
            
            # Resolve penetration
            penetration_vec = self.player_pos - closest_point
            if penetration_vec.length_squared() > 0:
                self.player_pos = closest_point + penetration_vec.normalize() * self.RIDER_RADIUS

            # Apply forces along the slope
            line_angle_rad = math.atan2(line_vec.y, line_vec.x)
            gravity_force = self.GRAVITY * math.sin(line_angle_rad)
            
            current_speed = self.player_vel.dot(line_vec.normalize())
            new_speed = current_speed + gravity_force
            self.player_vel = line_vec.normalize() * new_speed
            
            self.player_pos += self.player_vel
            self.player_angle = -math.degrees(line_angle_rad)

        # Create speed particles
        if self.player_vel.length() > 3:
            for _ in range(2):
                self.particles.append({
                    'pos': self.player_pos.copy() + pygame.Vector2(self.np_random.uniform(-5, 5), self.np_random.uniform(-5, 5)),
                    'vel': -self.player_vel.normalize() * self.np_random.uniform(1, 3),
                    'life': self.np_random.integers(10, 20),
                    'radius': self.np_random.uniform(1, 3),
                    'color': (255, 255, 100) # Yellow
                })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _update_camera(self):
        target_y = self.player_pos.y - self.HEIGHT / 2.5
        self.camera_y += (target_y - self.camera_y) * 0.1

    def _calculate_reward(self, last_x_pos, was_on_ground):
        reward = 0
        # Forward movement reward
        if self.player_pos.x > last_x_pos:
            reward += 0.1
        else:
            reward -= 0.1
        
        # High-speed jump reward
        if was_on_ground and not self.on_ground and self.player_vel.length() > self.JUMP_REWARD_SPEED_THRESHOLD:
            reward += 2.0
            # sfx_jump
            
        return reward

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
            
        if self.player_pos.y > self.HEIGHT + self.RIDER_RADIUS * 5 or self.player_pos.x < -self.RIDER_RADIUS * 5:
            self.crashed = True
            return True
            
        if self.player_pos.x >= self.finish_line_x:
            self.finished = True
            return True
            
        return False

    def _get_observation(self):
        # Clear screen with background gradient
        self.screen.fill(self.COLOR_BG_TOP)
        bottom_rect = pygame.Rect(0, self.HEIGHT / 2, self.WIDTH, self.HEIGHT / 2)
        pygame.draw.rect(self.screen, self.COLOR_BG_BOTTOM, bottom_rect)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_offset = pygame.Vector2(0, self.camera_y)

        # Render track lines
        all_lines = self.world_lines + self.player_lines
        for p1, p2 in all_lines:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1 - cam_offset, p2 - cam_offset)
        
        # Render finish line
        finish_start = pygame.Vector2(self.finish_line_x, 0)
        finish_end = pygame.Vector2(self.finish_line_x, self.HEIGHT)
        pygame.draw.line(self.screen, self.COLOR_FINISH, finish_start - cam_offset, finish_end - cam_offset, 3)

        # Render particles
        for p in self.particles:
            pos = p['pos'] - cam_offset
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(p['radius']), p['color'])

        # Render player
        player_screen_pos = self.player_pos - cam_offset
        int_pos = (int(player_screen_pos.x), int(player_screen_pos.y))
        
        # Glow effect
        glow_radius = int(self.RIDER_RADIUS * 1.8)
        s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER_GLOW)
        self.screen.blit(s, (int_pos[0] - glow_radius, int_pos[1] - glow_radius))

        # Rotated sled
        p1 = pygame.Vector2(-self.RIDER_RADIUS, 0).rotate(self.player_angle)
        p2 = pygame.Vector2(self.RIDER_RADIUS, 0).rotate(self.player_angle)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, int_pos+p1, int_pos+p2, 4)
        pygame.gfxdraw.filled_circle(self.screen, int_pos[0], int_pos[1], 4, self.COLOR_PLAYER)

        # Render cursor and anchor
        pygame.gfxdraw.filled_circle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), 5, self.COLOR_CURSOR)
        pygame.draw.circle(self.screen, self.COLOR_CURSOR, (int(self.cursor_pos.x), int(self.cursor_pos.y)), 8, 1)
        pygame.draw.line(self.screen, self.COLOR_ANCHOR, self.line_anchor_pos - cam_offset, self.cursor_pos, 1)
    
    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        if self.game_over:
            msg = "FINISH!" if self.finished else "CRASHED!"
            end_text = self.font_large.render(msg, True, self.COLOR_FINISH if self.finished else self.COLOR_PLAYER)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crashed": self.crashed,
            "finished": self.finished
        }
    
    def _create_crash_explosion(self):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 8)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': self.player_pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(20, 40),
                'radius': self.np_random.uniform(2, 5),
                'color': (255, self.np_random.integers(100, 200), 0) # Orange/Yellow
            })

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Set auto_advance to False for manual play to control frame rate
    env.auto_advance = False 
    
    while running:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # In manual mode, we only step when an action is taken or to advance physics
        # For this game, we want continuous physics, so we always step.
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render to screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # If you have a display, you can view the game
        try:
            display_screen = pygame.display.get_surface()
            if display_screen is None:
                display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
        except pygame.error: # Headless mode
            pass

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        env.clock.tick(30) # Control FPS for manual play

    pygame.quit()