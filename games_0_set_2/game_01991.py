
# Generated: 2025-08-28T03:22:22.326278
# Source Brief: brief_01991.md
# Brief Index: 1991

        
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
        "Controls: ↑↓←→ to aim the track. Press Space to build."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a sledder to the finish line by drawing track segments. "
        "Create a smooth path to build speed, but be careful not to crash!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.COLOR_BG = (135, 206, 235)  # Sky Blue
        self.COLOR_TERRAIN = (100, 149, 237)  # Cornflower Blue
        self.COLOR_TRACK = (40, 40, 40)
        self.COLOR_PREVIEW = (105, 105, 105) # Dim Gray
        self.COLOR_SLEDDER = (255, 255, 255)
        self.COLOR_START = (0, 200, 0)
        self.COLOR_FINISH = (200, 0, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_SHADOW = (0, 0, 0, 100)

        self.GRAVITY = 0.15
        self.FRICTION = 0.998
        self.TRACK_SEGMENT_LENGTH = 50
        self.MAX_STEPS = 1000
        self.FINISH_LINE_X = self.WIDTH - 50

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # State variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.sledder_pos = None
        self.sledder_vel = None
        self.track_segments = []
        self.terrain_points = []
        self.last_track_endpoint = None
        self.preview_angle = 0.0
        self.last_space_held = False
        self.particles = []
        self.max_dist_reached = 0.0

        # Initialize state by calling reset
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.sledder_pos = pygame.Vector2(50, 100)
        self.sledder_vel = pygame.Vector2(1, 0)

        start_platform_p1 = pygame.Vector2(10, 120)
        start_platform_p2 = pygame.Vector2(70, 120)
        self.track_segments = [(start_platform_p1, start_platform_p2)]
        self.last_track_endpoint = pygame.Vector2(start_platform_p2)

        self.preview_angle = 0.0
        self.last_space_held = False
        self.particles = []
        
        self._generate_terrain()
        
        self.max_dist_reached = self.sledder_pos.x

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # 1. Handle player input for track building
        self._handle_input(movement, space_held)

        # 2. Update game physics and calculate reward
        reward = self._update_physics()
        self.score += reward

        # 3. Update visual effects
        self._update_particles()

        # 4. Check for termination
        self.steps += 1
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.sledder_pos.x >= self.FINISH_LINE_X:
                win_reward = 100
                reward += win_reward
                self.score += win_reward
            else:
                crash_penalty = -10
                # Overwrite step reward with penalty, but keep total score accumulation
                self.score = self.score - reward + crash_penalty
                reward = crash_penalty
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        angle_change_speed = 0.08
        # 1=up, 2=down, 3=left, 4=right
        if movement in [1, 4]:  # Up or Right -> rotate Counter-Clockwise
            self.preview_angle -= angle_change_speed
        if movement in [2, 3]:  # Down or Left -> rotate Clockwise
            self.preview_angle += angle_change_speed
        
        self.preview_angle = np.clip(self.preview_angle, -math.pi / 2.1, math.pi / 2.1)

        if space_held and not self.last_space_held:
            # sfx_place_track
            p1 = self.last_track_endpoint
            p2 = p1 + pygame.Vector2(
                self.TRACK_SEGMENT_LENGTH * math.cos(self.preview_angle),
                self.TRACK_SEGMENT_LENGTH * math.sin(self.preview_angle),
            )
            self.track_segments.append((p1, p2))
            self.last_track_endpoint = p2
        
        self.last_space_held = space_held

    def _update_physics(self):
        prev_x = self.sledder_pos.x
        
        self.sledder_vel.y += self.GRAVITY

        is_on_track = False
        for p1, p2 in reversed(self.track_segments):
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
            
            p_to_s = self.sledder_pos - p1
            t = p_to_s.dot(line_vec) / line_vec.length_squared()
            
            if 0 <= t <= 1:
                closest_point = p1 + t * line_vec
                dist_vec = self.sledder_pos - closest_point
                
                if dist_vec.length() < 8 and self.sledder_pos.y <= closest_point.y + 3:
                    is_on_track = True
                    self.sledder_pos = closest_point
                    
                    track_tangent = line_vec.normalize()
                    projected_vel_scalar = self.sledder_vel.dot(track_tangent)
                    self.sledder_vel = track_tangent * projected_vel_scalar
                    
                    self.sledder_vel *= self.FRICTION
                    
                    if abs(track_tangent.angle_to(pygame.Vector2(1, 0))) > 10 and self.sledder_vel.length() > 1:
                        self._create_particles(self.sledder_pos)
                    break

        if not is_on_track:
            # sfx_freefall_whoosh
            terrain_y = self._get_terrain_height_at(self.sledder_pos.x)
            if self.sledder_pos.y >= terrain_y:
                # sfx_terrain_scrape
                self.sledder_pos.y = terrain_y
                self.sledder_vel.x *= 0.9
                if self.sledder_vel.y > 0: self.sledder_vel.y = 0

        self.sledder_pos += self.sledder_vel
        
        reward = 0
        if self.sledder_pos.x > self.max_dist_reached:
            reward = (self.sledder_pos.x - self.max_dist_reached) * 0.1
            self.max_dist_reached = self.sledder_pos.x
        
        return reward

    def _check_termination(self):
        if self.sledder_pos.x >= self.FINISH_LINE_X:
            # sfx_win
            return True
        if not (0 < self.sledder_pos.y < self.HEIGHT and -10 < self.sledder_pos.x < self.WIDTH + 10):
            # sfx_crash
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _generate_terrain(self):
        self.terrain_points = []
        y = self.HEIGHT * 0.8
        x = -50
        slope = self.np_random.uniform(0.05, 0.15)
        
        while x < self.WIDTH + 50:
            self.terrain_points.append(pygame.Vector2(x, y))
            slope += self.np_random.uniform(-0.05, 0.05)
            slope = np.clip(slope, -0.3, 0.3)
            y += slope * 10
            y = np.clip(y, self.HEIGHT * 0.5, self.HEIGHT - 20)
            x += 10

    def _get_terrain_height_at(self, x_pos):
        for i in range(len(self.terrain_points) - 1):
            p1 = self.terrain_points[i]
            p2 = self.terrain_points[i+1]
            if p1.x <= x_pos < p2.x:
                if p2.x == p1.x: return p1.y
                t = (x_pos - p1.x) / (p2.x - p1.x)
                return p1.y + t * (p2.y - p1.y)
        return self.HEIGHT

    def _create_particles(self, pos):
        num_particles = self.np_random.integers(1, 4)
        for _ in range(num_particles):
            p_vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-0.5, -2))
            self.particles.append({
                'pos': pos.copy() + pygame.Vector2(0, 3),
                'vel': p_vel,
                'life': self.np_random.integers(15, 25),
                'max_life': 25,
                'size': self.np_random.uniform(1.5, 3.5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.9
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        if self.terrain_points:
            poly_points = self.terrain_points + [(self.WIDTH, self.HEIGHT), (0, self.HEIGHT)]
            pygame.draw.polygon(self.screen, self.COLOR_TERRAIN, [p for p in poly_points])

        pygame.draw.line(self.screen, self.COLOR_START, (70, 0), (70, self.HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.FINISH_LINE_X, 0), (self.FINISH_LINE_X, self.HEIGHT), 3)

        for p1, p2 in self.track_segments:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1, p2)
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, (p1.x, p1.y+1), (p2.x, p2.y+1))

        if not self.game_over:
            p1 = self.last_track_endpoint
            p2 = p1 + pygame.Vector2(
                self.TRACK_SEGMENT_LENGTH * math.cos(self.preview_angle),
                self.TRACK_SEGMENT_LENGTH * math.sin(self.preview_angle),
            )
            self._draw_dashed_line(self.screen, self.COLOR_PREVIEW, p1, p2, 5)

        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (255, 255, 255, alpha)
            size = max(0, int(p['size'] * (p['life'] / p['max_life'])))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), size, color)

        s_pos = (int(self.sledder_pos.x), int(self.sledder_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, s_pos[0], s_pos[1], 8, (255, 255, 255, 50))
        pygame.gfxdraw.aacircle(self.screen, s_pos[0], s_pos[1], 8, (255, 255, 255, 100))
        pygame.draw.circle(self.screen, self.COLOR_SLEDDER, s_pos, 5)

    def _render_ui(self):
        dist_text = f"Distance: {int(self.max_dist_reached)} / {self.FINISH_LINE_X}"
        score_text = f"Score: {int(self.score)}"
        
        dist_surf = self.font.render(dist_text, True, self.COLOR_TEXT)
        dist_shadow = self.font.render(dist_text, True, self.COLOR_SHADOW)
        self.screen.blit(dist_shadow, (11, 11))
        self.screen.blit(dist_surf, (10, 10))

        score_surf = self.font.render(score_text, True, self.COLOR_TEXT)
        score_shadow = self.font.render(score_text, True, self.COLOR_SHADOW)
        self.screen.blit(score_shadow, (11, 41))
        self.screen.blit(score_surf, (10, 40))
        
        if self.game_over:
            status_text = ""
            if self.sledder_pos.x >= self.FINISH_LINE_X:
                status_text = "FINISH!"
                color = self.COLOR_START
            else:
                status_text = "CRASHED"
                color = self.COLOR_FINISH
            
            status_surf = self.font.render(status_text, True, color)
            shadow_surf = self.font.render(status_text, True, self.COLOR_SHADOW)
            text_rect = status_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(shadow_surf, text_rect.move(2, 2))
            self.screen.blit(status_surf, text_rect)

    def _draw_dashed_line(self, surf, color, start_pos, end_pos, dash_length=5):
        origin = pygame.Vector2(start_pos)
        target = pygame.Vector2(end_pos)
        displacement = target - origin
        length = displacement.length()
        if length == 0: return
        
        n_dashes = int(length / dash_length)
        
        for i in range(n_dashes):
            if i % 2 == 0:
                start = origin + displacement * (i / n_dashes)
                end = origin + displacement * ((i + 0.5) / n_dashes)
                pygame.draw.aaline(surf, color, start, end)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance": self.max_dist_reached,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        pygame_events = pygame.event.get()
        for event in pygame_events:
            if event.type == pygame.QUIT:
                running = False

        # Map keyboard inputs to action space
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Distance: {info['distance']}")
            obs, info = env.reset() # Auto-reset

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        pygame.display.get_surface().blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control frame rate
        env.clock.tick(60)

    env.close()