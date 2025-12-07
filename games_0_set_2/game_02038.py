
# Generated: 2025-08-27T19:03:31.932092
# Source Brief: brief_02038.md
# Brief Index: 2038

        
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
        "Controls: ↑ to rotate paddle clockwise, ↓ to rotate counter-clockwise. "
        "Angle your paddle to add spin to the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Curveball Pong: an isometric sports game where angled paddles bend the ball's "
        "trajectory to outmaneuver your opponent. First to 5 points wins."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 64)
        self.font_small = pygame.font.Font(None, 32)
        self.font_tiny = pygame.font.Font(None, 20)

        # --- Game Constants ---
        self.FPS = 30
        self.MAX_SCORE = 5
        self.MAX_CONSECUTIVE_MISSES = 2
        self.MAX_EPISODE_STEPS = 1000

        # --- Visuals & Colors ---
        self.COLOR_BG = (13, 13, 38) # Dark Blue
        self.COLOR_COURT = (25, 25, 77)
        self.COLOR_LINES = (200, 200, 255)
        self.COLOR_NET = (150, 150, 200, 150)
        self.COLOR_PLAYER = (0, 255, 255) # Cyan
        self.COLOR_OPPONENT = (255, 0, 255) # Magenta
        self.COLOR_BALL = (255, 255, 0) # Yellow
        self.COLOR_UI_TEXT = (230, 230, 255)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)
        
        # --- Court & Isometric Projection ---
        self.court_logical_width = 120
        self.court_logical_height = 240
        self.origin_x = self.screen_width // 2
        self.origin_y = 60
        
        # --- Game State (initialized in reset) ---
        self.rng = None
        self.steps = 0
        self.player_score = 0
        self.opponent_score = 0
        self.player_consecutive_misses = 0
        self.game_over = False
        self.game_over_message = ""
        self.game_phase = "serve" # 'serve', 'play', 'scored'
        self.phase_timer = 0
        
        self.ball = {}
        self.player_paddle = {}
        self.opponent_paddle = {}
        
        self.particles = []
        self.ball_trail = []
        self.serve_direction = 1 # 1 for player, -1 for opponent
        
        self.opponent_reaction_time = 0.2
        self.opponent_reaction_timer = 0.0
        self.opponent_target_angle = 0.0

        # Initialize state variables and validate
        self.reset()
        self.validate_implementation()

    def _iso_transform(self, x, y):
        """Converts logical court coordinates to screen coordinates."""
        screen_x = self.origin_x + (x - y) * 0.9
        screen_y = self.origin_y + (x + y) * 0.45
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.player_score = 0
        self.opponent_score = 0
        self.player_consecutive_misses = 0
        self.game_over = False
        self.game_over_message = ""
        
        self.player_paddle = {
            'y': 20, 'angle': 0, 'rot_speed': math.radians(6), 'width': 40
        }
        self.opponent_paddle = {
            'y': self.court_logical_height - 20, 'angle': 0, 'width': 40
        }
        
        self.opponent_reaction_time = 0.2
        self.particles = []
        self.ball_trail = []
        
        self._serve_ball()
        
        return self._get_observation(), self._get_info()

    def _serve_ball(self):
        self.game_phase = "serve"
        self.phase_timer = self.FPS // 2 # 0.5 second pause
        self.ball_trail.clear()
        
        ball_speed = 4.5
        
        if self.serve_direction == 1: # Player serves
            y_pos = self.player_paddle['y'] + 15
            vel_y = ball_speed
        else: # Opponent serves
            y_pos = self.opponent_paddle['y'] - 15
            vel_y = -ball_speed
            
        self.ball = {
            'pos': np.array([self.court_logical_width / 2, y_pos], dtype=float),
            'vel': np.array([0, vel_y], dtype=float),
            'spin': np.array([0, 0], dtype=float), # Spin is a persistent acceleration
            'radius': 5
        }

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        
        if not self.game_over:
            movement = action[0]
            
            if self.game_phase == 'play':
                self._update_player_paddle(movement)
                self._update_opponent_ai()
                reward += self._update_ball()
            elif self.game_phase in ['serve', 'scored']:
                self.phase_timer -= 1
                if self.phase_timer <= 0:
                    self.game_phase = 'play'

        # Check for termination conditions
        terminated = self._check_termination()
        
        # Apply terminal rewards
        if terminated:
            if self.player_score >= self.MAX_SCORE:
                reward += 10.0
            elif self.player_consecutive_misses >= self.MAX_CONSECUTIVE_MISSES:
                reward -= 10.0

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player_paddle(self, movement):
        # action[0]: 1=up (CW), 2=down (CCW)
        if movement == 1:
            self.player_paddle['angle'] += self.player_paddle['rot_speed']
        elif movement == 2:
            self.player_paddle['angle'] -= self.player_paddle['rot_speed']
        
        # Clamp angle to prevent full rotation
        max_angle = math.radians(75)
        self.player_paddle['angle'] = np.clip(self.player_paddle['angle'], -max_angle, max_angle)

    def _update_opponent_ai(self):
        # AI only acts if ball is in its half and moving towards it
        if self.ball['pos'][1] > self.court_logical_height / 2 and self.ball['vel'][1] > 0:
            self.opponent_reaction_timer -= 1 / self.FPS
            if self.opponent_reaction_timer <= 0:
                # Predict where ball will cross the paddle line
                time_to_impact = (self.opponent_paddle['y'] - self.ball['pos'][1]) / self.ball['vel'][1]
                if time_to_impact > 0:
                    predicted_x = self.ball['pos'][0] + self.ball['vel'][0] * time_to_impact
                    # Add some randomness to the target
                    predicted_x += (self.rng.random() - 0.5) * 15
                    
                    # Ideal angle to hit towards center of player's side
                    target_x = self.court_logical_width / 2
                    dx = target_x - predicted_x
                    dy = self.player_paddle['y'] - self.opponent_paddle['y']
                    
                    # This is a simplified angle calculation for returning the ball
                    # A more direct approach: aim to make the paddle perpendicular to the desired ball path
                    ideal_angle = math.atan2(dx, -dy) - math.pi/2
                    
                    # Add random variation to the angle for less predictable returns
                    ideal_angle += (self.rng.random() - 0.5) * math.radians(20)
                    self.opponent_target_angle = np.clip(ideal_angle, -math.radians(75), math.radians(75))

                self.opponent_reaction_timer = self.opponent_reaction_time

        # Smoothly move paddle towards target angle
        lerp_rate = 0.15
        self.opponent_paddle['angle'] += (self.opponent_target_angle - self.opponent_paddle['angle']) * lerp_rate

    def _update_ball(self):
        reward = 0
        
        # Apply spin and decay
        self.ball['vel'] += self.ball['spin']
        self.ball['spin'] *= 0.95 # Spin decays over time

        # Update ball position
        self.ball['pos'] += self.ball['vel']
        
        # Add to trail
        self.ball_trail.append(self.ball['pos'].copy())
        if len(self.ball_trail) > 15:
            self.ball_trail.pop(0)

        # Anti-softlock: if ball is too slow, reset serve
        if np.linalg.norm(self.ball['vel']) < 1.0:
            self.serve_direction *= -1
            self._serve_ball()
            return -1 # Penalize for softlock state

        # --- Collision Detection ---
        pos, vel, radius = self.ball['pos'], self.ball['vel'], self.ball['radius']

        # Side walls
        if (pos[0] < radius and vel[0] < 0) or (pos[0] > self.court_logical_width - radius and vel[0] > 0):
            vel[0] *= -1
            self.ball['spin'][0] *= -0.5 # Dampen spin on wall hit
            self._create_particles(pos, 5)
            # sfx: wall_bounce.wav

        # Scoring
        if pos[1] < 0: # Opponent scores
            self.opponent_score += 1
            self.player_consecutive_misses += 1
            self.serve_direction = -1 # Opponent serves next
            self._serve_ball()
            self.game_phase = "scored"
            # sfx: score_opponent.wav
            return -1.0
        if pos[1] > self.court_logical_height: # Player scores
            self.player_score += 1
            self.player_consecutive_misses = 0
            self.opponent_reaction_time = max(0.05, self.opponent_reaction_time - 0.005) # Opponent gets faster
            self.serve_direction = 1 # Player serves next
            self._serve_ball()
            self.game_phase = "scored"
            # sfx: score_player.wav
            return 1.0

        # Paddle collisions
        # Player
        if vel[1] < 0 and self.player_paddle['y'] < pos[1] < self.player_paddle['y'] + 15:
            paddle_x_center = self.court_logical_width / 2
            paddle_half_width = self.player_paddle['width'] / 2
            if paddle_x_center - paddle_half_width < pos[0] < paddle_x_center + paddle_half_width:
                reward += self._handle_paddle_collision(self.player_paddle)
                # sfx: hit_player.wav
        
        # Opponent
        if vel[1] > 0 and self.opponent_paddle['y'] - 15 < pos[1] < self.opponent_paddle['y']:
            paddle_x_center = self.court_logical_width / 2
            paddle_half_width = self.opponent_paddle['width'] / 2
            if paddle_x_center - paddle_half_width < pos[0] < paddle_x_center + paddle_half_width:
                self._handle_paddle_collision(self.opponent_paddle)
                # sfx: hit_opponent.wav
        
        # Small penalty for ball being in player's half
        if pos[1] < self.court_logical_height / 2:
            reward -= 0.01

        return reward

    def _handle_paddle_collision(self, paddle):
        # Reflect velocity
        self.ball['vel'][1] *= -1.05 # Add a bit of speed on return
        self.ball['vel'][0] += math.sin(paddle['angle']) * 4.0 # Horizontal impulse from angle
        
        # Add spin based on paddle angle
        # This is the "curveball" mechanic
        spin_amount = math.sin(paddle['angle']) * 0.15
        self.ball['spin'] = np.array([spin_amount, 0], dtype=float)
        
        # Clamp max speed
        speed = np.linalg.norm(self.ball['vel'])
        if speed > 10:
            self.ball['vel'] = self.ball['vel'] / speed * 10
        
        self._create_particles(self.ball['pos'], 15, is_player_paddle=(paddle == self.player_paddle))
        
        # Reward for successful return by player
        return 0.1 if paddle == self.player_paddle else 0


    def _check_termination(self):
        if self.game_over:
            return True
        
        if self.player_score >= self.MAX_SCORE:
            self.game_over = True
            self.game_over_message = "YOU WIN!"
            return True
        if self.player_consecutive_misses >= self.MAX_CONSECUTIVE_MISSES or self.opponent_score >= self.MAX_SCORE:
            self.game_over = True
            self.game_over_message = "YOU LOSE"
            return True
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            self.game_over_message = "TIME UP"
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
            "score": self.player_score,
            "opponent_score": self.opponent_score,
            "steps": self.steps,
        }

    def _render_game(self):
        self._draw_court()
        self._update_and_draw_particles()
        self._draw_ball_and_trail()
        self._draw_paddle(self.player_paddle, self.COLOR_PLAYER)
        self._draw_paddle(self.opponent_paddle, self.COLOR_OPPONENT)

    def _draw_court(self):
        w, h = self.court_logical_width, self.court_logical_height
        
        # Court floor polygon
        points = [self._iso_transform(0, 0), self._iso_transform(w, 0), 
                  self._iso_transform(w, h), self._iso_transform(0, h)]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_COURT)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_LINES)
        
        # Net
        net_y = h / 2
        start_pos = self._iso_transform(0, net_y)
        end_pos = self._iso_transform(w, net_y)
        pygame.draw.line(self.screen, self.COLOR_NET, start_pos, end_pos, 2)
        
    def _draw_paddle(self, paddle, color):
        center_x = self.court_logical_width / 2
        center_y = paddle['y']
        angle = paddle['angle']
        half_width = paddle['width'] / 2

        x1 = center_x - half_width * math.cos(angle)
        y1_offset = -half_width * math.sin(angle)
        
        x2 = center_x + half_width * math.cos(angle)
        y2_offset = half_width * math.sin(angle)
        
        start_pos = self._iso_transform(x1, center_y + y1_offset)
        end_pos = self._iso_transform(x2, center_y + y2_offset)

        pygame.draw.line(self.screen, color, start_pos, end_pos, 8)
        # Glow effect
        pygame.draw.line(self.screen, (255, 255, 255, 50), start_pos, end_pos, 12)

    def _draw_ball_and_trail(self):
        # Trail
        for i, pos in enumerate(self.ball_trail):
            if i % 2 == 0: # Draw every other point for a dashed look
                alpha = int(255 * (i / len(self.ball_trail)))
                color = (*self.COLOR_BALL, alpha)
                screen_pos = self._iso_transform(pos[0], pos[1])
                radius = int(self.ball['radius'] * 0.5 * (i / len(self.ball_trail)))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius, color)

        # Ball
        ball_pos = self.ball['pos']
        screen_pos = self._iso_transform(ball_pos[0], ball_pos[1])
        radius = self.ball['radius']
        
        # Simple shadow
        shadow_pos = self._iso_transform(ball_pos[0], ball_pos[1] + 5)
        pygame.gfxdraw.filled_circle(self.screen, shadow_pos[0], shadow_pos[1], radius, (0,0,0,100))
        
        # Ball itself with glow
        pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], radius, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius+3, (*self.COLOR_BALL, 50))

    def _create_particles(self, logical_pos, count, is_player_paddle=False):
        screen_pos = self._iso_transform(logical_pos[0], logical_pos[1])
        color = self.COLOR_PLAYER if is_player_paddle else self.COLOR_OPPONENT if not is_player_paddle else self.COLOR_LINES
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 2 + 1
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.rng.integers(10, 20)
            radius = self.rng.random() * 2 + 1
            self.particles.append({'pos': list(screen_pos), 'vel': velocity, 'life': lifetime, 'max_life': lifetime, 'radius': radius, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / p['max_life']))
                color = (*p['color'], alpha)
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                radius = int(p['radius'] * (p['life'] / p['max_life']))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _render_ui(self):
        # Score display
        player_text = self.font_large.render(str(self.player_score), True, self.COLOR_PLAYER)
        opponent_text = self.font_large.render(str(self.opponent_score), True, self.COLOR_OPPONENT)
        
        self.screen.blit(player_text, (self.screen_width / 2 - player_text.get_width() - 20, 10))
        self.screen.blit(opponent_text, (self.screen_width / 2 + 20, 10))
        
        # Serve indicator
        if self.game_phase == 'serve':
            text = "SERVE"
            surf = self.font_small.render(text, True, self.COLOR_UI_TEXT)
            pos_y = self.screen_height / 2 - 40 if self.serve_direction == 1 else self.screen_height / 2 + 20
            self.screen.blit(surf, (self.screen_width/2 - surf.get_width()/2, pos_y))

        # Game over message
        if self.game_over:
            color = self.COLOR_WIN if self.player_score >= self.MAX_SCORE else self.COLOR_LOSE
            text_surf = self.font_large.render(self.game_over_message, True, color)
            text_rect = text_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 150))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(text_surf, text_rect)
            
    def close(self):
        pygame.quit()
        
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Curveball Pong")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action mapping for human keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1 # Clockwise
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2 # Counter-clockwise
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}-{info['opponent_score']}")
            # The environment will show the game over screen, but we wait for 'r' to reset.
        
        # --- Rendering to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()