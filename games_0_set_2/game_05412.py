
# Generated: 2025-08-28T04:55:34.336302
# Source Brief: brief_05412.md
# Brief Index: 5412

        
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
        "Controls: Use arrow keys to draw the next piece of track. ↑ for up, ↓ for down, → for flat. Get the sled to the green finish line!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw a track in real-time for your sled to race on. Design the perfect path to get to the finish line before you crash or run out of time."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.W, self.H = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)
        
        # Colors
        self.COLOR_BG = (190, 210, 250)
        self.COLOR_TRACK = (255, 255, 255)
        self.COLOR_SLED = (220, 30, 30)
        self.COLOR_FINISH = (30, 220, 30)
        self.COLOR_UI_TEXT = (10, 10, 50)
        self.COLOR_PARTICLE_BASE = (230, 230, 255)

        # Game constants
        self.GRAVITY = 0.25
        self.FRICTION = 0.995
        self.MAX_STEPS = 400
        self.FINISH_LINE_X = self.W - 40
        self.TRACK_SEGMENT_LENGTH = 25.0
        
        # Initialize state variables
        self.sled_pos = None
        self.sled_vel = None
        self.sled_angle = None
        self.track_points = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.sled_pos = np.array([50.0, 200.0])
        self.sled_vel = np.array([2.0, 0.0])
        self.sled_angle = 0.0
        
        # Initial flat track segment
        self.track_points = [[0, 200], [100, 200]]
        
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1  # Not used in this game
        # shift_held = action[2] == 1 # Not used in this game
        
        old_sled_pos_x = self.sled_pos[0]

        if not self.game_over:
            # 1. Update track based on player action
            self._update_track(movement)
            # 2. Update sled physics
            self._update_physics()
            # 3. Update visual effects
            self._update_particles()
        
        self.steps += 1
        
        # 4. Check for termination conditions
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            
        # 5. Calculate reward
        reward = self._calculate_reward(old_sled_pos_x, terminated)
        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_track(self, movement):
        last_point = np.array(self.track_points[-1])
        angle = 0

        # Map movement action to an angle for the new track segment
        if movement == 0 or movement == 4:  # none or right -> flat
            angle = 0
        elif movement == 1:  # up
            angle = math.radians(30)
        elif movement == 2:  # down
            angle = math.radians(-30)
        elif movement == 3:  # left -> draw backwards (discouraged)
            angle = math.radians(180)

        dx = self.TRACK_SEGMENT_LENGTH * math.cos(angle)
        dy = -self.TRACK_SEGMENT_LENGTH * math.sin(angle)  # Pygame Y is inverted
        
        new_point = last_point + np.array([dx, dy])
        
        # Clamp new point to be within screen bounds
        new_point[0] = np.clip(new_point[0], 0, self.W)
        new_point[1] = np.clip(new_point[1], 0, self.H)

        self.track_points.append(new_point.tolist())
        
        # Limit total track points to prevent performance/memory issues
        if len(self.track_points) > 150:
            self.track_points.pop(1)  # Keep the very first point

    def _update_physics(self):
        # Apply gravity
        self.sled_vel[1] += self.GRAVITY

        # Tentative move based on velocity
        self.sled_pos += self.sled_vel

        # Collision detection and response with the track
        min_dist = float('inf')
        closest_seg_info = None

        for i in range(len(self.track_points) - 1):
            p1 = np.array(self.track_points[i])
            p2 = np.array(self.track_points[i+1])
            
            line_vec = p2 - p1
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0: continue

            p_to_sled = self.sled_pos - p1
            t = np.dot(p_to_sled, line_vec) / line_len_sq
            t = np.clip(t, 0, 1)
            
            closest_pt = p1 + t * line_vec
            dist = np.linalg.norm(self.sled_pos - closest_pt)

            if dist < min_dist:
                min_dist = dist
                closest_seg_info = {'p1': p1, 'p2': p2, 'closest_pt': closest_pt}

        on_track = False
        if closest_seg_info and min_dist < 15:
            # Sled is close enough to a track segment
            p1, p2, closest_pt = closest_seg_info['p1'], closest_seg_info['p2'], closest_seg_info['closest_pt']
            
            # Snap position to the track
            self.sled_pos = closest_pt.copy()
            on_track = True
            
            # Calculate track angle and normal vector
            seg_vec = p2 - p1
            self.sled_angle = math.atan2(-seg_vec[1], seg_vec[0]) if np.linalg.norm(seg_vec) > 0 else 0
            normal = np.array([math.sin(self.sled_angle), math.cos(self.sled_angle)])
            
            # Project velocity along track surface
            dot_product = np.dot(self.sled_vel, normal)
            self.sled_vel -= normal * dot_product
            
            # Apply friction
            self.sled_vel *= self.FRICTION
            
            # # SFX: Snow scraping sound
            self._spawn_particles()
        
        if not on_track:
            # Sled is in the air
            self.sled_angle = math.atan2(self.sled_vel[1], self.sled_vel[0])

    def _spawn_particles(self):
        if np.linalg.norm(self.sled_vel) > 1 and self.np_random.random() < 0.8:
            num_particles = self.np_random.integers(1, 4)
            for _ in range(num_particles):
                # Eject particles opposite to sled's velocity with some randomness
                particle_vel = -self.sled_vel * self.np_random.uniform(0.1, 0.4)
                particle_vel += self.np_random.uniform(-0.5, 0.5, 2)
                self.particles.append({
                    'pos': self.sled_pos.copy(),
                    'vel': particle_vel,
                    'life': self.np_random.integers(15, 25),
                    'max_life': 25,
                    'size': self.np_random.uniform(1, 4)
                })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_termination(self):
        win = self.sled_pos[0] >= self.FINISH_LINE_X
        crash = not (0 <= self.sled_pos[0] < self.W and 0 <= self.sled_pos[1] < self.H)
        timeout = self.steps >= self.MAX_STEPS
        return win or crash or timeout

    def _calculate_reward(self, old_sled_pos_x, terminated):
        reward = 0
        
        # Reward for forward progress
        dx = self.sled_pos[0] - old_sled_pos_x
        reward += dx * 0.1
        
        # Small survival reward
        if not terminated:
            reward += 0.01

        if terminated:
            if self.sled_pos[0] >= self.FINISH_LINE_X:  # Win
                reward += 100
                # # SFX: Win jingle
            else:  # Crash or timeout
                reward -= 10
                # # SFX: Crash sound
                
        return reward
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw checkered finish line for better visibility
        for i in range(0, self.H, 20):
            color1 = self.COLOR_FINISH if (i // 20) % 2 == 0 else (20, 150, 20)
            color2 = (20, 150, 20) if (i // 20) % 2 == 0 else self.COLOR_FINISH
            pygame.draw.rect(self.screen, color1, (self.FINISH_LINE_X, i, 10, 10))
            pygame.draw.rect(self.screen, color2, (self.FINISH_LINE_X, i + 10, 10, 10))

        # Draw track with a subtle shadow for depth
        if len(self.track_points) > 1:
            shadow_points = [(p[0] + 2, p[1] + 2) for p in self.track_points]
            pygame.draw.aalines(self.screen, (0, 0, 0, 50), False, shadow_points, 7)
            pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, self.track_points, 5)

        # Draw particles
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            alpha = int(150 * life_ratio)
            color = (*self.COLOR_PARTICLE_BASE, alpha)
            size = int(p['size'] * life_ratio)
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

        # Draw sled
        self._render_sled()

    def _render_sled(self):
        sled_len, sled_width = 20, 8
        cos_a, sin_a = math.cos(self.sled_angle), math.sin(self.sled_angle)
        
        # Use a rotated rectangle for the sled body
        rel_points = [
            np.array([-sled_len/2, -sled_width/2]), np.array([sled_len/2, -sled_width/2]),
            np.array([sled_len/2, sled_width/2]), np.array([-sled_len/2, sled_width/2]),
        ]
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        abs_points = [self.sled_pos + rot_matrix @ p for p in rel_points]
        int_points = [(int(p[0]), int(p[1])) for p in abs_points]

        pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_SLED)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_SLED)

    def _render_ui(self):
        score_text = f"Score: {self.score:.1f}"
        steps_text = f"Actions: {self.steps}/{self.MAX_STEPS}"
        
        score_surf = self.font.render(score_text, True, self.COLOR_UI_TEXT)
        steps_surf = self.font.render(steps_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(steps_surf, (self.W - steps_surf.get_width() - 10, 10))
        
        if self.game_over:
            status = ""
            if self.sled_pos[0] >= self.FINISH_LINE_X:
                status = "FINISH!"
            elif not (0 <= self.sled_pos[0] < self.W and 0 <= self.sled_pos[1] < self.H):
                status = "CRASHED!"
            else:
                status = "OUT OF ACTIONS!"
            
            status_surf = self.font.render(status, True, self.COLOR_UI_TEXT)
            self.screen.blit(status_surf, (self.W/2 - status_surf.get_width()/2, self.H/2 - status_surf.get_height()/2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # Example of how to use the environment
    env = GameEnv()
    env.validate_implementation()
    
    # --- Manual Play ---
    # This part is for human testing and is not part of the required deliverable
    obs, info = env.reset()
    done = False
    
    # Pygame window for display
    display_screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Sled Drawer")
    
    action = np.array([0, 0, 0]) # Start with no-op
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
        # Map keyboard to MultiDiscrete action space
        keys = pygame.key.get_pressed()
        move_action = 0 # none
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4

        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([move_action, space_action, shift_action])
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we need to control the "speed" of human input
        pygame.time.wait(50) # wait 50ms between actions
        
    print(f"Game Over. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    
    env.close()