
# Generated: 2025-08-27T20:40:07.930214
# Source Brief: brief_02536.md
# Brief Index: 2536

        
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
        "Controls: Arrows to aim track. Space to draw a boost track. Shift to draw a slowdown track."
    )

    game_description = (
        "Draw dynamic tracks to guide a physics-based rider to the finish line. Use different track types to control speed and trajectory."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WORLD_WIDTH = 5000

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)

        # Colors
        self.COLOR_BG = (15, 18, 26)
        self.COLOR_RIDER = (255, 255, 255)
        self.COLOR_TRACK_NORMAL = (46, 196, 182)
        self.COLOR_TRACK_BOOST = (51, 153, 255)
        self.COLOR_TRACK_SLOW = (255, 107, 107)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_FINISH = (253, 255, 150)
        
        # Game constants
        self.GRAVITY = pygame.Vector2(0, 0.15)
        self.RIDER_RADIUS = 7
        self.BOUNCE_FACTOR = 0.6
        self.FINISH_X = self.WORLD_WIDTH - 200
        self.MAX_TIME = 100.0
        self.MAX_STEPS = 3000 # 100 seconds at 30fps
        self.ACTION_INTERVAL = 5 # Draw a track every 5 steps
        self.TRACK_LENGTH = 35
        
        # Reward constants
        self.REWARD_FINISH = 100.0
        self.REWARD_CRASH = -100.0
        self.REWARD_PROGRESS = 0.1
        self.REWARD_BOOST_USE = 1.0
        self.REWARD_SLOW_USE = -0.5
        
        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""
        self.timer = self.MAX_TIME
        
        # Rider state
        self.rider_pos = pygame.Vector2(100, 200)
        self.rider_vel = pygame.Vector2(2, 0)
        self.prev_rider_x = self.rider_pos.x

        # Track state
        self.tracks = deque(maxlen=200) # Store recent track segments
        self.last_track_point = pygame.Vector2(0, 250)
        self._create_initial_track()

        # Camera
        self.camera_y = self.rider_pos.y - self.SCREEN_HEIGHT / 2

        # Effects
        self.particles = []
        self.step_events = [] # To store one-off reward events

        return self._get_observation(), self._get_info()

    def _create_initial_track(self):
        p1 = self.last_track_point
        for i in range(5):
            p2 = p1 + pygame.Vector2(self.TRACK_LENGTH, 0)
            self._add_track_segment(p1, p2, 'normal')
            p1 = p2
        self.last_track_point = p1

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # --- Action Handling ---
        if self.steps % self.ACTION_INTERVAL == 0:
            self._handle_action(action)

        # --- Game Logic Update ---
        self._update_physics()
        self._update_camera()
        self._update_particles()
        self.timer -= 1 / 30.0 # Assuming 30 FPS
        self.steps += 1
        
        # --- Reward Calculation ---
        reward = self._calculate_reward()
        self.score += reward
        
        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.termination_reason == "finish":
                reward += self.REWARD_FINISH
                self.score += self.REWARD_FINISH
            else:
                reward += self.REWARD_CRASH
                self.score += self.REWARD_CRASH
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action
        p1 = self.last_track_point
        angle_change = 0.6
        
        direction_vector = pygame.Vector2(1, 0) # Default: straight
        if movement == 1: # Up
            direction_vector = pygame.Vector2(1, -angle_change).normalize()
        elif movement == 2: # Down
            direction_vector = pygame.Vector2(1, angle_change).normalize()
        elif movement == 3: # Left (short forward)
            direction_vector = pygame.Vector2(0.7, 0)
        elif movement == 4: # Right (long forward)
            direction_vector = pygame.Vector2(1.3, 0)
        
        p2 = p1 + direction_vector * self.TRACK_LENGTH

        track_type = 'normal'
        if space_held:
            track_type = 'boost'
        elif shift_held:
            track_type = 'slow'
            
        self._add_track_segment(p1, p2, track_type)
        self.last_track_point = p2

    def _add_track_segment(self, p1, p2, track_type):
        color_map = {
            'normal': self.COLOR_TRACK_NORMAL,
            'boost': self.COLOR_TRACK_BOOST,
            'slow': self.COLOR_TRACK_SLOW,
        }
        segment = {
            'p1': p1, 'p2': p2, 'type': track_type, 
            'color': color_map[track_type], 'used': False
        }
        self.tracks.append(segment)

    def _update_physics(self):
        # Apply gravity
        self.rider_vel += self.GRAVITY
        
        # Apply velocity
        self.rider_pos += self.rider_vel

        # Collision detection and response
        collided_this_step = False
        for track in self.tracks:
            if collided_this_step: break
            
            line_vec = track['p2'] - track['p1']
            if line_vec.length_squared() == 0: continue
            
            p1_to_rider = self.rider_pos - track['p1']
            t = p1_to_rider.dot(line_vec) / line_vec.length_squared()
            t = max(0, min(1, t)) # Clamp to line segment
            
            closest_pt = track['p1'] + t * line_vec
            dist_vec = self.rider_pos - closest_pt
            
            if dist_vec.length() < self.RIDER_RADIUS:
                collided_this_step = True
                
                # Resolve penetration
                self.rider_pos = closest_pt + dist_vec.normalize() * self.RIDER_RADIUS
                
                # Calculate impulse
                normal = dist_vec.normalize()
                vn = self.rider_vel.dot(normal)
                
                if vn < 0: # Moving towards the surface
                    # Reflect velocity
                    self.rider_vel -= (1 + self.BOUNCE_FACTOR) * vn * normal
                    
                    # Apply track effects
                    if not track['used']:
                        if track['type'] == 'boost':
                            self.rider_vel *= 1.2 # 20% speed increase
                            self.step_events.append('boost')
                            # sfx: boost sound
                            self._spawn_particles(self.rider_pos, self.COLOR_TRACK_BOOST, 20, 3)
                        elif track['type'] == 'slow':
                            self.rider_vel *= 0.8 # 20% speed decrease
                            self.step_events.append('slow')
                            # sfx: slowdown sound
                            self._spawn_particles(self.rider_pos, self.COLOR_TRACK_SLOW, 20, 2)
                        track['used'] = True
                
                # sfx: scrape/thump sound

    def _update_camera(self):
        target_cam_y = self.rider_pos.y - self.SCREEN_HEIGHT * 0.5
        # Smooth camera movement (lerp)
        self.camera_y += (target_cam_y - self.camera_y) * 0.1

    def _calculate_reward(self):
        reward = 0
        
        # Progress reward
        progress = self.rider_pos.x - self.prev_rider_x
        reward += progress * self.REWARD_PROGRESS
        self.prev_rider_x = self.rider_pos.x
        
        # Event-based rewards
        for event in self.step_events:
            if event == 'boost':
                reward += self.REWARD_BOOST_USE
            elif event == 'slow':
                reward += self.REWARD_SLOW_USE
        self.step_events.clear()
        
        return reward

    def _check_termination(self):
        # Reached finish line
        if self.rider_pos.x >= self.FINISH_X:
            self.termination_reason = "finish"
            return True
        
        # Fell out of world
        if self.rider_pos.y > self.camera_y + self.SCREEN_HEIGHT + 100 or self.rider_pos.y < self.camera_y - 100:
            self.termination_reason = "crash (bounds)"
            return True
            
        # Went too far back
        if self.rider_pos.x < -50:
            self.termination_reason = "crash (backward)"
            return True
            
        # Timer ran out
        if self.timer <= 0:
            self.termination_reason = "timeout"
            return True
        
        # Max steps reached
        if self.steps >= self.MAX_STEPS:
            self.termination_reason = "max_steps"
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_offset = pygame.Vector2(0, self.camera_y)

        # Draw finish line
        finish_p1 = (self.FINISH_X, self.camera_y - 200)
        finish_p2 = (self.FINISH_X, self.camera_y + self.SCREEN_HEIGHT + 200)
        pygame.draw.line(self.screen, self.COLOR_FINISH, finish_p1, finish_p2, 3)

        # Draw tracks
        for track in self.tracks:
            p1_screen = track['p1'] - cam_offset
            p2_screen = track['p2'] - cam_offset
            pygame.draw.line(self.screen, track['color'], p1_screen, p2_screen, 5)

        # Draw particles
        for p in self.particles:
            p_screen = p['pos'] - cam_offset
            pygame.draw.circle(self.screen, p['color'], (int(p_screen.x), int(p_screen.y)), int(p['size']))

        # Draw rider
        rider_screen_pos = self.rider_pos - cam_offset
        rider_int_pos = (int(rider_screen_pos.x), int(rider_screen_pos.y))
        # Glow effect
        for i in range(self.RIDER_RADIUS, self.RIDER_RADIUS + 5):
            alpha = 100 - (i - self.RIDER_RADIUS) * 20
            color = (*self.COLOR_RIDER, alpha)
            pygame.gfxdraw.aacircle(self.screen, rider_int_pos[0], rider_int_pos[1], i, color)
        # Main circle
        pygame.gfxdraw.filled_circle(self.screen, rider_int_pos[0], rider_int_pos[1], self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, rider_int_pos[0], rider_int_pos[1], self.RIDER_RADIUS, self.COLOR_RIDER)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        timer_text = self.font_ui.render(f"TIME: {max(0, self.timer):.1f}", True, self.COLOR_UI)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            message = "GAME OVER"
            if self.termination_reason == "finish":
                message = "FINISH!"
            
            end_text = self.font_big.render(message, True, self.COLOR_FINISH)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rider_pos": (self.rider_pos.x, self.rider_pos.y),
            "rider_vel": (self.rider_vel.x, self.rider_vel.y),
            "timer": self.timer,
            "termination_reason": self.termination_reason
        }
        
    def _spawn_particles(self, pos, color, count, speed_factor):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_factor
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': random.uniform(10, 20),
                'size': random.uniform(1, 4),
                'color': color,
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['size'] > 0]

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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # To run with manual controls, uncomment the following block
    # and ensure you have pygame installed (`pip install pygame`)
    """
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation() # Run self-check
    obs, info = env.reset()
    done = False
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Line Rider Gym Environment")
    clock = pygame.time.Clock()
    
    while not done:
        # Map keyboard to MultiDiscrete action
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()
    """

    # --- To test with random actions (default execution) ---
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    episodes = 3
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
        print(f"Episode {ep+1}: Steps={step_count}, Score={info['score']:.2f}, Reason='{info['termination_reason']}'")
    env.close()