
# Generated: 2025-08-27T20:03:50.081069
# Source Brief: brief_02330.md
# Brief Index: 2330

        
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
        "Controls: ↑/↓/←/→ keys modify the drawing angle of the next track segment. "
        "Hold Shift to draw a longer segment. Press Space to place a speed boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A physics-based puzzle game. Draw a track for the rider to navigate from the start to the finish. "
        "Use speed boosts and careful angles to build a fast and successful path."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        # Colors
        self.COLOR_BG = (40, 42, 54)
        self.COLOR_GRID = (68, 71, 90)
        self.COLOR_TRACK = (248, 248, 242)
        self.COLOR_RIDER = (255, 85, 85)
        self.COLOR_BOOST = (255, 249, 128)
        self.COLOR_START = (80, 250, 123)
        self.COLOR_FINISH = (255, 121, 198)
        self.COLOR_TEXT = (248, 248, 242)

        # Game and Physics Parameters
        self.RIDER_RADIUS = 8
        self.GRAVITY = 0.15
        self.FRICTION = 0.995 # Multiplier for tangential velocity
        self.BOUNCINESS = 0.4 # Restitution coefficient
        self.SEGMENT_LENGTH_NORMAL = 25
        self.SEGMENT_LENGTH_LONG = 50
        self.BOOST_POWER = 7.0
        self.START_LINE_X = 60
        self.FINISH_LINE_X = self.WIDTH - 60
        self.START_POS = pygame.math.Vector2(self.START_LINE_X, self.HEIGHT / 3)
        self.MAX_BOOSTS = 5
        self.MAX_STEPS = 400 # Max number of line segments to draw
        self.PHYSICS_SUBSTEPS = 20 # Physics simulation steps per agent action
        self.STUCK_VEL_THRESHOLD_SQ = 0.1**2
        self.STUCK_FRAMES_LIMIT = 80

        # Reward structure
        self.REWARD_PROGRESS = 0.05
        self.REWARD_BOOST = 5.0
        self.REWARD_WIN = 100.0
        self.REWARD_LOSS = -100.0
        self.REWARD_STUCK = -100.0
        self.REWARD_TIMEOUT = -20.0

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_desc = pygame.font.SysFont("Consolas", 14)
        except pygame.error:
            self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
            self.font_desc = pygame.font.SysFont("monospace", 14)


        # --- Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # State variables will be initialized in reset()
        self.rider_pos = None
        self.rider_vel = None
        self.track_segments = None
        self.last_track_point = None
        self.current_angle = None
        self.boosts = None
        self.boosts_remaining = None
        self.stuck_counter = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.np_random = None

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.rider_pos = self.START_POS.copy()
        self.rider_vel = pygame.math.Vector2(2.0, 0)
        
        # Initial flat track segment
        p1 = pygame.math.Vector2(self.START_LINE_X - 40, self.START_POS.y)
        p2 = pygame.math.Vector2(self.START_POS.x + 20, self.START_POS.y)
        self.track_segments = [(p1, p2)]
        self.last_track_point = p2
        self.current_angle = 0.0

        self.boosts = []
        self.boosts_remaining = self.MAX_BOOSTS
        
        self.stuck_counter = 0
        self.steps = 0
        self.score = 0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Interpret action and modify the track
        self._handle_action(action)
        self.steps += 1

        # 2. Run physics simulation
        reward, terminated_in_sim = self._run_physics_simulation()
        self.score += reward

        # 3. Check for other termination conditions
        terminated = terminated_in_sim
        if not terminated:
            if self.steps >= self.MAX_STEPS:
                terminated = True
                self.score += self.REWARD_TIMEOUT
                reward += self.REWARD_TIMEOUT

        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Action 0 (movement) modifies the current drawing angle
        angle_change = 0
        if movement == 0: pass # no-op, continue straight
        elif movement == 1: angle_change = -15  # Gentle Up
        elif movement == 2: angle_change = 15   # Gentle Down
        elif movement == 3: angle_change = -35  # Sharp Up
        elif movement == 4: angle_change = 35   # Sharp Down
        self.current_angle += angle_change
        self.current_angle = max(-85, min(85, self.current_angle)) # Clamp angle

        # Action 2 (shift) determines segment length
        length = self.SEGMENT_LENGTH_LONG if shift_held else self.SEGMENT_LENGTH_NORMAL

        # Create new segment
        angle_rad = math.radians(self.current_angle)
        end_point = self.last_track_point + pygame.math.Vector2(length * math.cos(angle_rad), length * math.sin(angle_rad))
        
        # Prevent drawing backwards
        if end_point.x < self.last_track_point.x:
            end_point.x = self.last_track_point.x + 1
        
        new_segment = (self.last_track_point, end_point)
        self.track_segments.append(new_segment)
        self.last_track_point = end_point

        # Action 1 (space) places a boost
        if space_held and self.boosts_remaining > 0:
            boost_pos = self.last_track_point.lerp(end_point, 0.5)
            self.boosts.append({"pos": boost_pos, "active": True})
            self.boosts_remaining -= 1
            # sfx: boost_placed.wav

    def _run_physics_simulation(self):
        total_reward = 0.0
        terminated = False

        for _ in range(self.PHYSICS_SUBSTEPS):
            old_pos_x = self.rider_pos.x
            
            # Apply gravity
            self.rider_vel.y += self.GRAVITY

            # Handle collisions and friction
            self._handle_collisions()

            # Update position
            self.rider_pos += self.rider_vel

            # Check for boost collection
            for boost in self.boosts:
                if boost["active"] and (boost["pos"] - self.rider_pos).length() < self.RIDER_RADIUS + 4:
                    boost["active"] = False
                    self.rider_vel = self.rider_vel.normalize() * (self.rider_vel.length() + self.BOOST_POWER)
                    total_reward += self.REWARD_BOOST
                    # sfx: boost_collect.wav

            # Reward for forward progress
            progress = self.rider_pos.x - old_pos_x
            if progress > 0:
                total_reward += progress * self.REWARD_PROGRESS

            # Check win/loss conditions
            if self.rider_pos.x >= self.FINISH_LINE_X:
                total_reward += self.REWARD_WIN
                terminated = True
                # sfx: win.wav
                break
            
            if self.rider_pos.y > self.HEIGHT + self.RIDER_RADIUS * 5:
                total_reward += self.REWARD_LOSS
                terminated = True
                # sfx: fall.wav
                break
            
            # Check if stuck
            if self.rider_vel.length_squared() < self.STUCK_VEL_THRESHOLD_SQ:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
            
            if self.stuck_counter >= self.STUCK_FRAMES_LIMIT:
                total_reward += self.REWARD_STUCK
                terminated = True
                # sfx: stuck.wav
                break
        
        return total_reward, terminated

    def _handle_collisions(self):
        is_on_track = False
        for p1, p2 in self.track_segments:
            # Simple broad-phase check
            rider_box = pygame.Rect(self.rider_pos.x - self.RIDER_RADIUS, self.rider_pos.y - self.RIDER_RADIUS, self.RIDER_RADIUS*2, self.RIDER_RADIUS*2)
            min_x, max_x = min(p1.x, p2.x), max(p1.x, p2.x)
            min_y, max_y = min(p1.y, p2.y), max(p1.y, p2.y)
            line_box = pygame.Rect(min_x - self.RIDER_RADIUS, min_y - self.RIDER_RADIUS, (max_x - min_x) + self.RIDER_RADIUS*2, (max_y - min_y) + self.RIDER_RADIUS*2)
            if not rider_box.colliderect(line_box):
                continue

            # Detailed narrow-phase check
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
            
            point_vec = self.rider_pos - p1
            t = point_vec.dot(line_vec) / line_vec.length_squared()
            t = max(0, min(1, t))
            
            closest_point = p1 + t * line_vec
            dist_vec = self.rider_pos - closest_point
            dist_sq = dist_vec.length_squared()

            if dist_sq < self.RIDER_RADIUS**2:
                is_on_track = True
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 1e-6
                
                # 1. Positional Correction (resolve penetration)
                penetration = self.RIDER_RADIUS - dist
                normal = dist_vec / dist
                self.rider_pos += normal * penetration
                
                # 2. Velocity Response (bounce)
                vel_along_normal = self.rider_vel.dot(normal)
                if vel_along_normal < 0:
                    impulse = -(1 + self.BOUNCINESS) * vel_along_normal
                    self.rider_vel += normal * impulse
                    # sfx: bounce.wav (low volume)
                
        # Apply friction only if on a track
        if is_on_track:
            self.rider_vel *= self.FRICTION

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
            "boosts_remaining": self.boosts_remaining,
            "rider_speed": self.rider_vel.length() if self.rider_vel else 0
        }

    def _render_game(self):
        self._render_grid()
        self._render_start_finish_lines()
        self._render_track()
        self._render_boosts()
        self._render_rider()

    def _render_grid(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_start_finish_lines(self):
        pygame.draw.line(self.screen, self.COLOR_START, (self.START_LINE_X, 0), (self.START_LINE_X, self.HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.FINISH_LINE_X, 0), (self.FINISH_LINE_X, self.HEIGHT), 3)

    def _render_track(self):
        for p1, p2 in self.track_segments:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), 3)
            
    def _render_boosts(self):
        for boost in self.boosts:
            if boost["active"]:
                self._draw_star(boost["pos"], 6, self.COLOR_BOOST)

    def _draw_star(self, center, size, color):
        points = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            radius = size if i % 2 == 0 else size / 2.5
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            points.append((int(x), int(y)))
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_rider(self):
        pos = (int(self.rider_pos.x), int(self.rider_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.RIDER_RADIUS, self.COLOR_RIDER)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        boost_text = self.font_ui.render(f"BOOSTS: {self.boosts_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(boost_text, (self.WIDTH - boost_text.get_width() - 10, 10))
        
        steps_text = self.font_desc.render(f"SEGMENTS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 35))

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

if __name__ == "__main__":
    # This block allows you to run the game and play it manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen_human = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Line Rider Gym")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Initial render
    frame = env._get_observation()
    frame = np.transpose(frame, (1, 0, 2))
    surf = pygame.surfarray.make_surface(frame)
    screen_human.blit(surf, (0, 0))
    pygame.display.flip()
    
    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("Press R to reset.")
    print("="*30 + "\n")

    while running:
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Any key that corresponds to an action triggers a step
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE, pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    action_taken = True
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
                    print("\n--- Environment Reset ---")
                    action_taken = True # Force a redraw after reset

        if action_taken:
            # --- Action mapping for human player ---
            move = 0 # No-op
            space = 0
            shift = 0

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: move = 1
            elif keys[pygame.K_DOWN]: move = 2
            elif keys[pygame.K_LEFT]: move = 3
            elif keys[pygame.K_RIGHT]: move = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [move, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
            
            if terminated:
                print("--- Episode Finished --- (Press R to reset)")
                
            # Render the new state
            frame = obs
            frame = np.transpose(frame, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen_human.blit(surf, (0, 0))
            pygame.display.flip()

        clock.tick(30) # Limit frame rate

    env.close()