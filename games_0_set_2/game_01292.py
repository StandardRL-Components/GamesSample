import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ↑ and ↓ to move your paddle. "
        "Try to hit the ball past your opponent to score."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro pixel-art Pong game. Win by scoring 8 points in each of the 3 increasingly difficult stages. "
        "You have 3 lives for the entire game. A stage ends if time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_LIVES = 3
        self.POINTS_TO_WIN_STAGE = 8
        self.MAX_STAGES = 3
        self.STAGE_TIME_LIMIT_SECONDS = 60

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        if "SDL_VIDEODRIVER" not in os.environ:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # Colors
        self.COLOR_BG = (15, 15, 15)
        self.COLOR_PADDLE_PLAYER = (50, 255, 50)
        self.COLOR_PADDLE_OPPONENT = (255, 50, 50)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_WALL = (200, 200, 200)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)

        # Fonts
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        self.player_paddle = None
        self.opponent_paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.particles = []

        self.steps = 0
        self.total_episode_score = 0
        self.stage_score = 0
        self.lives = 0
        self.stage = 0
        self.stage_timer = 0
        self.game_over = False
        self.win = False

        self.ball_base_speed = 0
        self.opponent_ai_speed = 0

        # self.reset() is called by the test harness, no need to call it here.

    def _get_difficulty_settings(self, stage):
        if stage == 1:
            return 7.0, 0.1 # Ball speed pixels/frame, opponent reaction speed
        elif stage == 2:
            return 8.0, 0.15
        else: # Stage 3
            return 9.0, 0.2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_paddle = pygame.Rect(30, self.HEIGHT / 2 - 40, 12, 80)
        self.opponent_paddle = pygame.Rect(self.WIDTH - 30 - 12, self.HEIGHT / 2 - 40, 12, 80)
        
        self.steps = 0
        self.total_episode_score = 0
        self.lives = self.MAX_LIVES
        self.stage = 1
        self.game_over = False
        self.win = False
        self.particles = []

        self._start_stage()

        return self._get_observation(), self._get_info()

    def _start_stage(self):
        self.stage_score = 0
        self.stage_timer = self.STAGE_TIME_LIMIT_SECONDS * self.FPS
        self.ball_base_speed, self.opponent_ai_speed = self._get_difficulty_settings(self.stage)
        self._reset_ball(is_new_stage=True)
    
    def _reset_ball(self, player_wins_point=True, is_new_stage=False):
        self.ball_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        
        if is_new_stage:
             # Start ball towards player after a stage win
             angle = self.np_random.uniform(-math.pi / 6, math.pi / 6)
             direction = -1
        else:
            angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
            direction = -1 if player_wins_point else 1

        self.ball_vel = pygame.Vector2(
            self.ball_base_speed * math.cos(angle) * direction,
            self.ball_base_speed * math.sin(angle)
        )

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        # --- Action Handling ---
        movement = action[0]  # 0-4: none/up/down/left/right
        
        paddle_moved = False
        if movement == 1:  # Up
            self.player_paddle.y -= 10
            paddle_moved = True
        elif movement == 2:  # Down
            self.player_paddle.y += 10
            paddle_moved = True

        if paddle_moved:
            reward -= 0.001 # Small cost for moving, encouraging efficiency

        self.player_paddle.y = np.clip(self.player_paddle.y, 0, self.HEIGHT - self.player_paddle.height)

        # --- Game Logic ---
        self._update_opponent_ai()
        self._update_ball()
        
        # --- Scoring and State Changes ---
        ball_rect = pygame.Rect(self.ball_pos.x - 5, self.ball_pos.y - 5, 10, 10)
        
        # Ball collision with paddles
        if ball_rect.colliderect(self.player_paddle) and self.ball_vel.x < 0:
            reward += self._handle_paddle_collision(self.player_paddle, ball_rect)
            # sfx: paddle_hit
        elif ball_rect.colliderect(self.opponent_paddle) and self.ball_vel.x > 0:
            self._handle_paddle_collision(self.opponent_paddle, ball_rect)
            # sfx: opponent_hit

        # Ball out of bounds
        if self.ball_pos.x < 0: # Player misses
            self.lives -= 1
            reward -= 1.0
            # sfx: lose_point
            if self.lives <= 0:
                self.game_over = True
            else:
                self._reset_ball(player_wins_point=False)
        elif self.ball_pos.x > self.WIDTH: # Opponent misses
            self.stage_score += 1
            self.total_episode_score += 1
            reward += 1.0
            # sfx: score_point
            if self.stage_score >= self.POINTS_TO_WIN_STAGE:
                reward += self._advance_stage()
            else:
                self._reset_ball(player_wins_point=True)

        # --- Update timer and particles ---
        self.stage_timer -= 1
        self._update_particles()
        
        self.steps += 1
        
        # --- Termination Check ---
        if self.stage_timer <= 0 and not self.game_over:
            self.game_over = True
            # sfx: game_over
        
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_opponent_ai(self):
        target_y = self.ball_pos.y - self.opponent_paddle.height / 2
        movement = (target_y - self.opponent_paddle.y) * self.opponent_ai_speed
        self.opponent_paddle.y += movement
        self.opponent_paddle.y = np.clip(self.opponent_paddle.y, 0, self.HEIGHT - self.opponent_paddle.height)
    
    def _update_ball(self):
        self.ball_pos += self.ball_vel

        # Top/bottom wall collision
        if self.ball_pos.y <= 5 or self.ball_pos.y >= self.HEIGHT - 5:
            self.ball_vel.y *= -1
            self.ball_pos.y = np.clip(self.ball_pos.y, 5, self.HEIGHT - 5)
            self._create_particles(self.ball_pos, self.COLOR_WALL, 10)
            # sfx: wall_bounce

    def _handle_paddle_collision(self, paddle, ball_rect):
        intersect_y = paddle.centery - ball_rect.centery
        normalized_intersect = intersect_y / (paddle.height / 2)
        bounce_angle = normalized_intersect * (math.pi / 3) # Max 60 degrees

        # Invert horizontal velocity and apply new vertical velocity
        self.ball_vel.x *= -1.05 # Speed up slightly on hit
        self.ball_vel.y = -self.ball_base_speed * math.sin(bounce_angle)

        # Ensure ball doesn't get stuck in paddle
        self.ball_pos.x += self.ball_vel.x * 1.5
        
        particle_color = self.COLOR_PADDLE_PLAYER if paddle is self.player_paddle else self.COLOR_PADDLE_OPPONENT
        self._create_particles(self.ball_pos, particle_color, 20)
        return 0.1 # Reward for hitting the ball

    def _advance_stage(self):
        self.stage += 1
        if self.stage > self.MAX_STAGES:
            self.game_over = True
            self.win = True
            # sfx: game_win
            return 100.0 # Big win reward
        else:
            self._start_stage()
            # sfx: stage_clear
            return 10.0 # Stage clear reward

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            particle = {
                "pos": pygame.Vector2(pos),
                "vel": pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)),
                "life": self.np_random.integers(10, 20),
                "color": color
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text(self, text, font, color, center_pos, shadow=True):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=center_pos)
        if shadow:
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            shadow_rect = shadow_surf.get_rect(center=(center_pos[0] + 2, center_pos[1] + 2))
            self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _render_game(self):
        # Center line
        for y in range(10, self.HEIGHT - 10, 20):
            pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH / 2 - 2, y, 4, 10))
        
        # Paddles
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_PLAYER, self.player_paddle, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_OPPONENT, self.opponent_paddle, border_radius=3)
        
        # Ball
        if self.ball_pos:
            ball_rect = pygame.Rect(0,0,10,10)
            ball_rect.center = (int(self.ball_pos.x), int(self.ball_pos.y))
            pygame.draw.rect(self.screen, self.COLOR_BALL, ball_rect, border_radius=2)
        
        # Particles
        for p in self.particles:
            alpha = min(255, max(0, p["life"] * 15))
            color_with_alpha = (*p["color"], alpha)
            size = max(1, int(p["life"] / 4))
            rect = pygame.Rect(int(p["pos"].x - size/2), int(p["pos"].y - size/2), size, size)
            
            # Create a temporary surface for alpha blending
            shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, color_with_alpha, shape_surf.get_rect())
            self.screen.blit(shape_surf, rect)

    def _render_ui(self):
        # Score
        score_text = f"{self.stage_score}"
        self._render_text(score_text, self.font_large, self.COLOR_TEXT, (self.WIDTH / 2, 40))

        # Stage
        stage_text = f"STAGE {self.stage}"
        self._render_text(stage_text, self.font_medium, self.COLOR_TEXT, (self.WIDTH - 100, 30))

        # Timer
        time_left = max(0, self.stage_timer // self.FPS)
        timer_color = self.COLOR_TEXT if time_left > 10 else self.COLOR_PADDLE_OPPONENT
        timer_text = f"{time_left:02d}"
        self._render_text(timer_text, self.font_medium, timer_color, (self.WIDTH / 2, self.HEIGHT - 30))

        # Lives
        for i in range(self.lives):
            self._draw_heart((50 + i * 35, 30), 12, self.COLOR_PADDLE_OPPONENT)
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PADDLE_PLAYER if self.win else self.COLOR_PADDLE_OPPONENT
            self._render_text(msg, self.font_large, color, (self.WIDTH / 2, self.HEIGHT / 2 - 20))

    def _draw_heart(self, pos, size, color):
        x, y = pos
        points = [
            (x, y + size // 4),
            (x - size // 2, y - size // 4),
            (x - size // 2, y - size // 2),
            (x, y - size // 2),
            (x + size // 2, y - size // 2),
            (x + size // 2, y - size // 4),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _get_info(self):
        return {
            "score": self.total_episode_score,
            "steps": self.steps,
            "stage": self.stage,
            "lives": self.lives,
        }
        
    def close(self):
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

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to run headless
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    # To play manually, you MUST comment out the os.environ line above
    # and have a display available.
    render_mode = "human" if "SDL_VIDEODRIVER" not in os.environ else "rgb_array"
    
    env = GameEnv(render_mode="rgb_array")
    
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Retro Pong")
        
        obs, info = env.reset()
        done = False
        
        while not done:
            movement = 0 # No-op
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2

            action = [movement, 0, 0] # Action format for MultiDiscrete
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Convert observation back to a surface for display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            env.clock.tick(env.FPS)
            
    finally:
        env.close()