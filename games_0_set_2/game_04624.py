
# Generated: 2025-08-28T02:57:24.252300
# Source Brief: brief_04624.md
# Brief Index: 4624

        
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
        "Controls: ↑/↓ to move your paddle. Press space to swing and hit the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, retro arcade tennis game. Outmaneuver your opponent and win the match by scoring 3 sets."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000 * 3 # Allow for longer matches
        
        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_COURT = (40, 80, 60)
        self.COLOR_LINES = (200, 200, 200)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_OPPONENT = (255, 80, 80)
        self.COLOR_BALL = (255, 255, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)

        # Entity properties
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 10, 60
        self.BALL_SIZE = 10
        self.PLAYER_SPEED = 8
        self.BALL_SPEED_INITIAL = 6
        self.BALL_SPEED_MAX = 12
        self.SWING_DURATION = 5  # frames

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_y = 0
        self.opponent_y = 0
        self.ball_x, self.ball_y = 0, 0
        self.ball_vx, self.ball_vy = 0, 0
        self.current_ball_speed = 0
        self.player_score, self.opponent_score = 0, 0
        self.player_sets, self.opponent_sets = 0, 0
        self.player_swing_timer = 0
        self.opponent_swing_timer = 0
        self.opponent_reaction_time = 0.4 # Initial delay
        self.opponent_target_y = self.HEIGHT / 2
        self.particles = []
        
        self.validate_implementation()

    def _serve_ball(self, for_player):
        self.ball_x = self.WIDTH / 2
        self.ball_y = self.HEIGHT / 2
        self.current_ball_speed = self.BALL_SPEED_INITIAL
        
        angle = self.np_random.uniform(-math.pi / 6, math.pi / 6)
        self.ball_vx = self.current_ball_speed * math.cos(angle)
        self.ball_vy = self.current_ball_speed * math.sin(angle)
        
        if not for_player:
            self.ball_vx *= -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_y = self.HEIGHT / 2
        self.opponent_y = self.HEIGHT / 2
        
        self.player_score = 0
        self.opponent_score = 0
        self.player_sets = 0
        self.opponent_sets = 0
        
        self.player_swing_timer = 0
        self.opponent_swing_timer = 0
        self.particles = []

        self.opponent_reaction_time = 0.4
        
        self._serve_ball(for_player=self.np_random.choice([True, False]))
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Player Control ---
        moved = False
        if movement == 1:  # Up
            self.player_y -= self.PLAYER_SPEED
            moved = True
        elif movement == 2:  # Down
            self.player_y += self.PLAYER_SPEED
            moved = True
        
        if moved:
            reward -= 0.01

        self.player_y = np.clip(self.player_y, self.PADDLE_HEIGHT / 2, self.HEIGHT - self.PADDLE_HEIGHT / 2)
        
        if space_held and self.player_swing_timer == 0:
            self.player_swing_timer = self.SWING_DURATION
            # SFX: whoosh

        # --- Opponent AI ---
        if self.np_random.random() > self.opponent_reaction_time:
            self.opponent_target_y = self.ball_y
        
        # Simple interpolation for smooth AI movement
        self.opponent_y += (self.opponent_target_y - self.opponent_y) * 0.1
        self.opponent_y = np.clip(self.opponent_y, self.PADDLE_HEIGHT / 2, self.HEIGHT - self.PADDLE_HEIGHT / 2)
        
        opponent_paddle_rect = pygame.Rect(self.WIDTH - 20 - self.PADDLE_WIDTH, self.opponent_y - self.PADDLE_HEIGHT / 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        if opponent_paddle_rect.left - self.ball_x < 50 and self.ball_vx > 0:
            if self.opponent_swing_timer == 0:
                self.opponent_swing_timer = self.SWING_DURATION
                # SFX: whoosh_opponent

        # --- Ball Physics & Collisions ---
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Top/Bottom wall collision
        if self.ball_y - self.BALL_SIZE / 2 < 0 or self.ball_y + self.BALL_SIZE / 2 > self.HEIGHT:
            self.ball_vy *= -1
            self.ball_y = np.clip(self.ball_y, self.BALL_SIZE / 2, self.HEIGHT - self.BALL_SIZE / 2)
            self._create_particles(self.ball_x, self.ball_y, self.COLOR_LINES, 10)
            # SFX: bounce_wall

        # Paddle collision
        player_paddle_rect = pygame.Rect(20, self.player_y - self.PADDLE_HEIGHT / 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        ball_rect = pygame.Rect(self.ball_x - self.BALL_SIZE / 2, self.ball_y - self.BALL_SIZE / 2, self.BALL_SIZE, self.BALL_SIZE)

        # Player hit
        if self.ball_vx < 0 and player_paddle_rect.colliderect(ball_rect) and self.player_swing_timer > 0:
            self.ball_vx *= -1
            hit_offset = (self.ball_y - self.player_y) / (self.PADDLE_HEIGHT / 2)
            self.ball_vy += hit_offset * 3  # Add spin
            self.current_ball_speed = min(self.BALL_SPEED_MAX, self.current_ball_speed * 1.05)
            self._normalize_ball_velocity()
            reward += 0.1
            self._create_particles(self.ball_x, self.ball_y, self.COLOR_PLAYER, 20)
            # SFX: hit_player

        # Opponent hit
        if self.ball_vx > 0 and opponent_paddle_rect.colliderect(ball_rect) and self.opponent_swing_timer > 0:
            self.ball_vx *= -1
            hit_offset = (self.ball_y - self.opponent_y) / (self.PADDLE_HEIGHT / 2)
            self.ball_vy += hit_offset * 2
            self.current_ball_speed = min(self.BALL_SPEED_MAX, self.current_ball_speed * 1.02)
            self._normalize_ball_velocity()
            self._create_particles(self.ball_x, self.ball_y, self.COLOR_OPPONENT, 20)
            # SFX: hit_opponent

        # --- Scoring ---
        point_over = False
        if self.ball_x < 0:
            self.opponent_score += 1
            reward -= 1
            point_over = True
            set_reward, match_reward = self._check_set_and_match()
            reward += set_reward + match_reward
            if not self.game_over: self._serve_ball(for_player=True)
        elif self.ball_x > self.WIDTH:
            self.player_score += 1
            reward += 1
            point_over = True
            set_reward, match_reward = self._check_set_and_match()
            reward += set_reward + match_reward
            if not self.game_over: self._serve_ball(for_player=False)
        
        if point_over:
            self.score = self.player_score - self.opponent_score # Simple score for info

        # --- Update timers and particles ---
        self.player_swing_timer = max(0, self.player_swing_timer - 1)
        self.opponent_swing_timer = max(0, self.opponent_swing_timer - 1)
        self._update_particles()
        
        # --- Termination ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _normalize_ball_velocity(self):
        magnitude = math.sqrt(self.ball_vx**2 + self.ball_vy**2)
        if magnitude > 0:
            self.ball_vx = (self.ball_vx / magnitude) * self.current_ball_speed
            self.ball_vy = (self.ball_vy / magnitude) * self.current_ball_speed

    def _check_set_and_match(self):
        set_reward = 0
        match_reward = 0
        
        if self.player_score >= 6:
            self.player_sets += 1
            self.player_score, self.opponent_score = 0, 0
            set_reward = 10
            # SFX: set_win
            self.opponent_reaction_time = max(0.1, self.opponent_reaction_time - 0.02)
        elif self.opponent_score >= 6:
            self.opponent_sets += 1
            self.player_score, self.opponent_score = 0, 0
            set_reward = -10
            # SFX: set_lose
        
        if self.player_sets >= 3:
            self.game_over = True
            match_reward = 100
            # SFX: match_win
        elif self.opponent_sets >= 3:
            self.game_over = True
            match_reward = -100
            # SFX: match_lose
            
        return set_reward, match_reward

    def _get_observation(self):
        # --- Drawing ---
        # Background
        self.screen.fill(self.COLOR_COURT)
        pygame.draw.rect(self.screen, self.COLOR_BG, (0, 0, self.WIDTH, self.HEIGHT), 20)
        
        # Court Lines
        pygame.draw.line(self.screen, self.COLOR_LINES, (self.WIDTH / 2, 0), (self.WIDTH / 2, self.HEIGHT), 2)
        
        # Render particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))

        # Render paddles
        swing_offset_p = self.PADDLE_WIDTH / 2 if self.player_swing_timer > 0 else 0
        player_rect = pygame.Rect(20 - swing_offset_p, self.player_y - self.PADDLE_HEIGHT / 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        swing_offset_o = self.PADDLE_WIDTH / 2 if self.opponent_swing_timer > 0 else 0
        opponent_rect = pygame.Rect(self.WIDTH - 20 - self.PADDLE_WIDTH + swing_offset_o, self.opponent_y - self.PADDLE_HEIGHT / 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_OPPONENT, opponent_rect, border_radius=3)

        # Render ball with a glow
        ball_center = (int(self.ball_x), int(self.ball_y))
        for i in range(4, 0, -1):
            glow_color = (self.COLOR_BALL[0]//(i+1), self.COLOR_BALL[1]//(i+1), self.COLOR_BALL[2]//(i+1))
            pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], int(self.BALL_SIZE/2 + i*2), glow_color)
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], int(self.BALL_SIZE/2), self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_center[0], ball_center[1], int(self.BALL_SIZE/2), self.COLOR_BALL)
        
        # Render UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_ui(self):
        # Score display
        player_score_text = self.font_small.render(f"P1: {self.player_score}", True, self.COLOR_TEXT)
        self.screen.blit(player_score_text, (30, 10))
        
        opponent_score_text = self.font_small.render(f"P2: {self.opponent_score}", True, self.COLOR_TEXT)
        text_rect = opponent_score_text.get_rect(topright=(self.WIDTH - 30, 10))
        self.screen.blit(opponent_score_text, text_rect)
        
        # Set display
        sets_text = self.font_large.render(f"{self.player_sets} - {self.opponent_sets}", True, self.COLOR_TEXT)
        text_rect = sets_text.get_rect(center=(self.WIDTH / 2, 30))
        self.screen.blit(sets_text, text_rect)
        
        if self.game_over:
            if self.player_sets >= 3:
                end_text = self.font_large.render("YOU WIN!", True, self.COLOR_WIN)
            else:
                end_text = self.font_large.render("YOU LOSE", True, self.COLOR_LOSE)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_sets": self.player_sets,
            "opponent_sets": self.opponent_sets,
        }

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.uniform(2, 5),
                'lifespan': self.np_random.integers(10, 20),
                'color': color
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['radius'] -= 0.15
            if p['lifespan'] > 0 and p['radius'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Override Pygame display for direct play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    print(env.user_guide)
    
    total_reward = 0
    
    # Game loop
    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # Cap the frame rate
        env.clock.tick(30)

    print(f"Game Over! Final Score (Total Reward): {total_reward:.2f}")
    print(f"Info: {info}")
    
    pygame.quit()