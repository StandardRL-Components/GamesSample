
# Generated: 2025-08-28T05:23:53.509036
# Source Brief: brief_02615.md
# Brief Index: 2615

        
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
        "Controls: Use arrows to move the cursor. Hold Space to draw a track. Hold Shift to erase."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A physics-based sledding game. Draw lines to create a track for the rider to reach the finish line. Go for speed and a smooth ride!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        
        # Colors
        self.COLOR_BG = (210, 225, 240)
        self.COLOR_TRACK = (60, 60, 80)
        self.COLOR_PLAYER = (0, 0, 0)
        self.COLOR_START = (0, 200, 0, 150)
        self.COLOR_FINISH = (220, 0, 0, 150)
        self.COLOR_CURSOR = (255, 100, 0)
        self.COLOR_TRAIL = (50, 100, 255)
        self.COLOR_IMPACT = (255, 255, 255)
        self.COLOR_TEXT = (30, 30, 50)
        
        # Game constants
        self.FPS = 30
        self.MAX_STEPS = 1500 # 50 seconds at 30fps
        self.GRAVITY = 0.4
        self.FRICTION = 0.99
        self.CURSOR_SPEED = 8
        self.RIDER_RADIUS = 6
        self.ERASER_RADIUS = 20
        self.MAX_LINE_LENGTH = 150 # Prevents overly long lines
        self.STOP_VELOCITY_THRESHOLD = 0.1
        
        # Initialize state variables
        self.lines = []
        self.rider_pos = pygame.Vector2(0, 0)
        self.rider_vel = pygame.Vector2(0, 0)
        self.cursor_pos = pygame.Vector2(0, 0)
        self.prev_cursor_pos = pygame.Vector2(0, 0)
        self.particles = []
        self.trails = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.max_speed_this_episode = 0.0
        self.stopped_frames_count = 0
        self.prev_rider_x = 0
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.max_speed_this_episode = 0.0
        self.stopped_frames_count = 0

        self.start_pos = pygame.Vector2(80, self.HEIGHT - 60)
        self.finish_x = self.WIDTH - 80

        self.rider_pos = self.start_pos.copy()
        self.rider_vel = pygame.Vector2(1, 0) # Small initial push
        self.prev_rider_x = self.rider_pos.x

        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.prev_cursor_pos = self.cursor_pos.copy()

        # Initial ground line
        self.lines = [
            ((0, self.HEIGHT - 50), (self.WIDTH, self.HEIGHT - 50))
        ]
        
        self.particles.clear()
        self.trails.clear()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Handle input and update game objects
        self._handle_input(movement, space_held, shift_held)
        
        # Update physics
        impact_force = self._update_rider_physics()
        
        # Update visual effects
        self._update_effects()

        # Calculate rewards
        # 1. Time penalty
        reward -= 0.01 
        
        # 2. Forward movement reward
        forward_progress = self.rider_pos.x - self.prev_rider_x
        reward += forward_progress * 0.1
        self.prev_rider_x = self.rider_pos.x

        # 3. High impact penalty (bumpy ride)
        if impact_force > 1.0:
            reward -= min(impact_force * 0.2, 2.0) # sound: *thud*

        # 4. Max speed reward
        current_speed = self.rider_vel.length()
        if current_speed > self.max_speed_this_episode:
            reward += 5.0 # sound: *whoosh*
            self.max_speed_this_episode = current_speed
            
        self.score += reward
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if self.rider_pos.x >= self.finish_x:
                self.score += 100 # sound: *victory fanfare*
                reward += 100
            else:
                self.score -= 100 # sound: *fail sound*
                reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        cursor_move = pygame.Vector2(0, 0)
        if movement == 1: cursor_move.y = -1
        elif movement == 2: cursor_move.y = 1
        elif movement == 3: cursor_move.x = -1
        elif movement == 4: cursor_move.x = 1
        
        self.cursor_pos += cursor_move * self.CURSOR_SPEED
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

        # Draw or erase
        if space_held and not shift_held:
            # Draw line if cursor moved and line isn't too long
            if self.cursor_pos.distance_to(self.prev_cursor_pos) > 1:
                new_line = (
                    (int(self.prev_cursor_pos.x), int(self.prev_cursor_pos.y)),
                    (int(self.cursor_pos.x), int(self.cursor_pos.y))
                )
                self.lines.append(new_line)
                # sound: *pencil scratch*
        
        if shift_held:
            # Erase lines near cursor
            self.lines = [
                line for line in self.lines
                if not self._is_line_near_point(line, self.cursor_pos, self.ERASER_RADIUS)
            ]
            # sound: *erase swoosh*

        self.prev_cursor_pos = self.cursor_pos.copy()

    def _update_rider_physics(self):
        # Apply gravity
        self.rider_vel.y += self.GRAVITY
        
        # Store pre-update velocity to calculate impact
        vel_before_collision = self.rider_vel.copy()

        # Simple iterative collision response
        collided = False
        for _ in range(2): # 2 iterations for stability
            collision_info = self._find_collision()
            if collision_info:
                collided = True
                p1, p2, dist = collision_info
                
                # Reposition rider to be exactly on the line
                normal = pygame.Vector2(p2[1] - p1[1], p1[0] - p2[0]).normalize()
                self.rider_pos += normal * (self.RIDER_RADIUS - dist)

                # Project velocity onto the line tangent
                tangent = (pygame.Vector2(p2) - pygame.Vector2(p1)).normalize()
                vel_dot_tangent = self.rider_vel.dot(tangent)
                self.rider_vel = tangent * vel_dot_tangent
                
                # Apply friction
                self.rider_vel *= self.FRICTION
            else:
                break
        
        # Update position
        self.rider_pos += self.rider_vel

        # Create impact particles if collision happened
        impact_force = (vel_before_collision - self.rider_vel).length()
        if collided and impact_force > 3.0:
            self._create_particles(self.rider_pos, int(impact_force * 2), self.COLOR_IMPACT, 1, 3)
        
        return impact_force

    def _find_collision(self):
        candidate_lines = []
        for line in self.lines:
            p1, p2 = line
            dist_sq = self._dist_point_to_segment_sq(self.rider_pos, p1, p2)
            if dist_sq < self.RIDER_RADIUS**2:
                candidate_lines.append((dist_sq, line))
        
        if not candidate_lines:
            return None
        
        # Return the closest colliding line
        best_dist_sq, best_line = min(candidate_lines, key=lambda x: x[0])
        return best_line[0], best_line[1], math.sqrt(best_dist_sq)

    def _update_effects(self):
        # Update particles (move, shrink, fade)
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[0] += p[1] # pos += vel
            p[2] -= 0.1  # life -= 1

        # Update trails
        if self.steps % 2 == 0:
            self.trails.append([self.rider_pos.copy(), self.rider_vel.length() * 2, 20]) # pos, size, life
        self.trails = [t for t in self.trails if t[2] > 0]
        for t in self.trails:
            t[2] -= 1

    def _check_termination(self):
        # 1. Rider reached finish line
        if self.rider_pos.x >= self.finish_x:
            return True
        
        # 2. Rider is out of bounds
        if not (0 < self.rider_pos.x < self.WIDTH and -50 < self.rider_pos.y < self.HEIGHT):
            return True
            
        # 3. Rider has stopped moving
        if self.rider_vel.length() < self.STOP_VELOCITY_THRESHOLD:
            self.stopped_frames_count += 1
        else:
            self.stopped_frames_count = 0
        if self.stopped_frames_count > self.FPS * 2: # Stopped for 2 seconds
            return True
            
        # 4. Max steps reached
        if self.steps >= self.MAX_STEPS:
            return True
            
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
            "rider_speed": self.rider_vel.length()
        }

    def _render_game(self):
        # Draw start/finish gates
        pygame.gfxdraw.box(self.screen, (int(self.start_pos.x) - 5, 0, 10, self.HEIGHT), self.COLOR_START)
        pygame.gfxdraw.box(self.screen, (int(self.finish_x) - 5, 0, 10, self.HEIGHT), self.COLOR_FINISH)
        
        # Draw trails
        for pos, size, life in self.trails:
            alpha = max(0, int(life * 12))
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(size), (*self.COLOR_TRAIL, alpha))

        # Draw all track lines
        for p1, p2 in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1, p2, 3)

        # Draw particles
        for pos, vel, life in self.particles:
            radius = int(life * 2)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, self.COLOR_IMPACT)

        # Draw rider
        p = self.rider_pos
        r = self.RIDER_RADIUS
        # Sled
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (p.x - r*1.5, p.y + r), (p.x + r*1.5, p.y + r), 4)
        # Body
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (p.x, p.y + r), (p.x, p.y - r), 3)
        # Head
        pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y - r), int(r*0.8), self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(p.x), int(p.y - r), int(r*0.8), self.COLOR_PLAYER)

        # Draw cursor
        cx, cy = int(self.cursor_pos.x), int(self.cursor_pos.y)
        pygame.draw.circle(self.screen, self.COLOR_CURSOR, (cx, cy), 10, 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx-5, cy), (cx+5, cy), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx, cy-5), (cx, cy+5), 2)

    def _render_ui(self):
        speed_text = f"SPEED: {self.rider_vel.length():.1f}"
        time_text = f"TIME: {self.steps / self.FPS:.1f}s"
        score_text = f"SCORE: {self.score:.0f}"

        speed_surf = self.font_ui.render(speed_text, True, self.COLOR_TEXT)
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)

        self.screen.blit(speed_surf, (10, 10))
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))
        self.screen.blit(score_surf, (self.WIDTH // 2 - score_surf.get_width() // 2, 10))
    
    def _create_particles(self, pos, count, color, min_speed, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(min_speed, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = random.uniform(0.5, 1.5)
            self.particles.append([pos.copy(), vel, life])

    # --- Helper Math Functions ---
    @staticmethod
    def _dist_point_to_segment_sq(p, v, w):
        l2 = (v[0] - w[0])**2 + (v[1] - w[1])**2
        if l2 == 0.0: return (p[0] - v[0])**2 + (p[1] - v[1])**2
        t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
        proj_x = v[0] + t * (w[0] - v[0])
        proj_y = v[1] + t * (w[1] - v[1])
        return (p[0] - proj_x)**2 + (p[1] - proj_y)**2

    @staticmethod
    def _is_line_near_point(line, point, radius):
        dist_sq = GameEnv._dist_point_to_segment_sq(point, line[0], line[1])
        return dist_sq < radius**2

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game with keyboard controls
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Mapping keyboard keys to actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Line Rider Gym Environment")
    
    action = env.action_space.sample()
    action.fill(0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # No movement
        for key, move_action in key_map.items():
            if keys[key]:
                action[0] = move_action
                break # Prioritize first key found
                
        # Space and Shift
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(1000) # Pause for a second on reset
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(env.FPS)
        
    env.close()