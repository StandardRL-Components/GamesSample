
# Generated: 2025-08-27T14:57:48.955961
# Source Brief: brief_00846.md
# Brief Index: 846

        
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
        "Controls: ↑ to accelerate, ↓ to brake/reverse, ←→ to turn. "
        "Complete 3 laps before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down arcade racer on a procedurally generated track. "
        "Dodge obstacles and race against the clock in a neon-drenched world."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Screen dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('Consolas', 20, bold=True)
        self.font_large = pygame.font.SysFont('Consolas', 48, bold=True)
        
        # Colors (Tron-inspired)
        self.COLOR_BG = (10, 10, 25)
        self.COLOR_PLAYER = (255, 0, 50)
        self.COLOR_PLAYER_GLOW = (255, 0, 50)
        self.COLOR_OBSTACLE = (255, 150, 0)
        self.COLOR_TRACK = (220, 220, 255)
        self.COLOR_FINISH = (0, 255, 100)
        self.COLOR_CHECKPOINT = (0, 150, 255)
        self.COLOR_TEXT = (255, 255, 255)

        # Game constants
        self.FPS = 30
        self.MAX_STEPS = self.FPS * 60  # 60 seconds
        self.LAPS_TO_WIN = 3
        self.TRACK_WIDTH = 80
        
        # Player physics constants
        self.THRUST_POWER = 0.3
        self.TURN_SPEED = 0.05
        self.DRAG_COEFFICIENT = 0.97
        self.BRAKE_POWER = 0.94
        self.MAX_SPEED = 15

        # Initialize state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = 0.0
        self.player_radius = 10
        self.camera_offset = pygame.Vector2(0, 0)

        self.track_center_points = []
        self.track_segments_left = []
        self.track_segments_right = []
        self.obstacles = []
        self.checkpoints = []
        self.particles = []
        
        self.steps = 0
        self.score = 0.0
        self.laps = 0
        self.next_checkpoint_index = 0
        self.game_over = False
        self.game_won = False
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.laps = 0
        self.game_over = False
        self.game_won = False
        self.particles.clear()
        
        self._generate_track()
        
        start_point = self.track_center_points[0]
        next_point = self.track_center_points[1]
        
        self.player_pos = pygame.Vector2(start_point)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = math.atan2(next_point[1] - start_point[1], next_point[0] - start_point[0])
        self.next_checkpoint_index = 1
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.01  # Small reward for surviving
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            
            # Update player physics
            self._update_player(movement)
            
            # Check for checkpoint crossing
            reward += self._check_checkpoints()
            
            # Check for obstacle collision
            if self._check_collisions():
                reward = -100.0
                self.game_over = True
                self._create_explosion(self.player_pos, self.COLOR_OBSTACLE)
                # sfx: car crash
            
        self._update_particles()
        
        self.steps += 1
        
        # Check termination conditions
        if not self.game_over:
            if self.laps >= self.LAPS_TO_WIN:
                reward += 100.0  # Big reward for winning
                self.game_over = True
                self.game_won = True
                self._create_explosion(self.player_pos, self.COLOR_FINISH)
                # sfx: win jingle
            elif self.steps >= self.MAX_STEPS:
                reward = -50.0  # Penalty for running out of time
                self.game_over = True
                # sfx: timeout buzzer

        terminated = self.game_over
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_track(self):
        # Clear previous track data
        self.track_center_points.clear()
        self.track_segments_left.clear()
        self.track_segments_right.clear()
        self.obstacles.clear()
        self.checkpoints.clear()
        
        # Generate a procedural track loop
        num_points = 12
        center = pygame.Vector2(2000, 2000)
        min_radius, max_radius = 600, 800
        
        points = []
        for _ in range(num_points):
            angle = self.np_random.uniform(0, 2 * math.pi)
            radius = self.np_random.uniform(min_radius, max_radius)
            points.append(center + pygame.Vector2(math.cos(angle) * radius, math.sin(angle) * radius))
        
        # Sort points by angle to create a non-intersecting loop
        points.sort(key=lambda p: math.atan2(p.y - center.y, p.x - center.x))
        
        # Create a closed loop
        self.track_center_points = points + [points[0]]
        
        # Create track boundaries and checkpoints
        for i in range(len(self.track_center_points) - 1):
            p1 = pygame.Vector2(self.track_center_points[i])
            p2 = pygame.Vector2(self.track_center_points[i+1])
            
            direction = (p2 - p1).normalize()
            perp = pygame.Vector2(-direction.y, direction.x)
            
            self.track_segments_left.append(((p1 - perp * self.TRACK_WIDTH).xy, (p2 - perp * self.TRACK_WIDTH).xy))
            self.track_segments_right.append(((p1 + perp * self.TRACK_WIDTH).xy, (p2 + perp * self.TRACK_WIDTH).xy))

            # Checkpoints are perpendicular to the track direction
            self.checkpoints.append(((p2 - perp * self.TRACK_WIDTH).xy, (p2 + perp * self.TRACK_WIDTH).xy))
            
            # Place obstacles
            if i > 0 and self.np_random.random() < 0.7: # Don't place on start/finish line
                for _ in range(self.np_random.integers(1, 4)):
                    dist_from_center = self.np_random.uniform(self.TRACK_WIDTH * 0.2, self.TRACK_WIDTH * 0.9)
                    side = self.np_random.choice([-1, 1])
                    pos_on_segment = self.np_random.uniform(0.1, 0.9)
                    
                    obstacle_center = p1.lerp(p2, pos_on_segment) + perp * dist_from_center * side
                    size = self.np_random.integers(15, 25)
                    self.obstacles.append(pygame.Rect(obstacle_center.x - size/2, obstacle_center.y - size/2, size, size))

    def _update_player(self, movement):
        # 1. Turning
        if movement == 3:  # Left
            self.player_angle -= self.TURN_SPEED
        if movement == 4:  # Right
            self.player_angle += self.TURN_SPEED
        
        # 2. Acceleration/Braking
        thrust = pygame.Vector2(0, 0)
        if movement == 1:  # Up
            thrust = pygame.Vector2(math.cos(self.player_angle), math.sin(self.player_angle)) * self.THRUST_POWER
            # sfx: engine accelerate
        elif movement == 2: # Down
            self.player_vel *= self.BRAKE_POWER
            # sfx: brake screech
            
        self.player_vel += thrust
        
        # 3. Drag & Speed Limit
        if self.player_vel.length() > self.MAX_SPEED:
            self.player_vel.scale_to_length(self.MAX_SPEED)
        self.player_vel *= self.DRAG_COEFFICIENT

        # 4. Update Position
        self.player_pos += self.player_vel
        
    def _check_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x - self.player_radius, self.player_pos.y - self.player_radius, self.player_radius*2, self.player_radius*2)
        for obstacle in self.obstacles:
            if player_rect.colliderect(obstacle):
                return True
        return False
        
    def _check_checkpoints(self):
        reward = 0
        p1 = self.player_pos - self.player_vel
        p2 = self.player_pos
        
        checkpoint_line = self.checkpoints[self.next_checkpoint_index]
        
        if self._line_segment_intersection(p1.xy, p2.xy, checkpoint_line[0], checkpoint_line[1]):
            # sfx: checkpoint chime
            self.next_checkpoint_index += 1
            if self.next_checkpoint_index >= len(self.checkpoints):
                self.laps += 1
                self.next_checkpoint_index = 0
                reward = 50.0  # Lap completed
                self._create_explosion(self.player_pos, self.COLOR_CHECKPOINT, 15)
            else:
                reward = 20.0  # Checkpoint passed
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['vel'] *= 0.98

    def _create_explosion(self, pos, color, count=40):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'lifespan': self.np_random.integers(20, 40),
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _get_observation(self):
        self.camera_offset = self.player_pos - pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render track
        for seg in self.track_segments_left + self.track_segments_right:
            p1 = pygame.Vector2(seg[0]) - self.camera_offset
            p2 = pygame.Vector2(seg[1]) - self.camera_offset
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1, p2, 2)
            
        # Render checkpoints and finish line
        for i, cp in enumerate(self.checkpoints):
            color = self.COLOR_FINISH if i == 0 else self.COLOR_CHECKPOINT
            p1 = pygame.Vector2(cp[0]) - self.camera_offset
            p2 = pygame.Vector2(cp[1]) - self.camera_offset
            pygame.draw.aaline(self.screen, color, p1, p2, 3)

        # Render obstacles
        for obs in self.obstacles:
            obs_rect = obs.move(-self.camera_offset.x, -self.camera_offset.y)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_rect)
            
        # Render particles
        for p in self.particles:
            pos = p['pos'] - self.camera_offset
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 40))))
            color = (*p['color'], alpha)
            pygame.gfxdraw.box(self.screen, (int(pos.x), int(pos.y), p['size'], p['size']), color)
            
        # Render player (always at screen center)
        center_screen = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        # Glow effect
        for i in range(10, 0, -2):
            alpha = 50 - i * 5
            pygame.gfxdraw.filled_circle(self.screen, int(center_screen.x), int(center_screen.y), 
                                         int(self.player_radius + i), (*self.COLOR_PLAYER_GLOW, alpha))

        # Car body
        p1 = center_screen + pygame.Vector2(math.cos(self.player_angle), math.sin(self.player_angle)) * self.player_radius * 1.5
        p2 = center_screen + pygame.Vector2(math.cos(self.player_angle + 2.2), math.sin(self.player_angle + 2.2)) * self.player_radius
        p3 = center_screen + pygame.Vector2(math.cos(self.player_angle - 2.2), math.sin(self.player_angle - 2.2)) * self.player_radius
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

    def _render_ui(self):
        # Lap Counter
        lap_text = f"LAP: {min(self.laps + 1, self.LAPS_TO_WIN)} / {self.LAPS_TO_WIN}"
        self._draw_text(lap_text, (20, 20), self.font_small)
        
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"TIME: {time_left:.1f}"
        self._draw_text(time_text, (self.SCREEN_WIDTH - 120, 20), self.font_small)

        # Score
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 40), self.font_small, center=True)
        
        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_FINISH if self.game_won else self.COLOR_PLAYER
            self._draw_text(msg, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.font_large, color=color, center=True)

    def _draw_text(self, text, pos, font, color=None, center=False):
        if color is None:
            color = self.COLOR_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laps": self.laps,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def _line_segment_intersection(self, p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return False
            
        t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))
        
        t = t_num / den
        u = u_num / den
        
        return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0
        
    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we need a different setup
    pygame.display.set_caption("Arcade Racer")
    real_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    obs, info = env.reset()
    terminated = False
    
    # Mapping keyboard keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while not terminated:
        # --- Human Controls ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        for key, move_action in key_to_action.items():
            if keys[key]:
                movement = move_action
                break # Prioritize up/down/left/right in this order
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    pygame.quit()