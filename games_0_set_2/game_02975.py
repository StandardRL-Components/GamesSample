import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑/↓ to move your paddle. Return the ball to score rounds. "
        "First to 5 rounds wins!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pixel Pong is a fast-paced isometric arcade game. "
        "Face off against an AI opponent and prove your reflexes."
    )

    # Frames auto-advance at 30fps.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.LOGICAL_WIDTH, self.LOGICAL_HEIGHT = 500, 250
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.ROUNDS_TO_WIN = 5

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts
        try:
            self.font_big = pygame.font.Font(pygame.font.get_default_font(), 48)
            self.font_medium = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
        except IOError:
            self.font_big = pygame.font.SysFont("monospace", 48)
            self.font_medium = pygame.font.SysFont("monospace", 24)
            self.font_small = pygame.font.SysFont("monospace", 16)

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_COURT = (40, 80, 60)
        self.COLOR_COURT_LINE = (180, 220, 200)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_ACCENT = (150, 220, 255)
        self.COLOR_AI = (255, 100, 100)
        self.COLOR_AI_ACCENT = (255, 180, 180)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_SHADOW = (0, 0, 0, 50)
        self.COLOR_TEXT = (255, 255, 200)

        # Game constants
        self.PADDLE_LOGICAL_HEIGHT = 60
        self.PADDLE_THICKNESS = 10
        self.PADDLE_VISUAL_HEIGHT = 20
        self.PLAYER_PADDLE_X = 40
        self.AI_PADDLE_X = self.LOGICAL_WIDTH - 40
        self.PLAYER_SPEED = 8
        self.BASE_AI_SPEED = 2.5
        self.BALL_RADIUS = 6
        self.INITIAL_BALL_SPEED = 6

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_round_wins = 0
        self.ai_round_wins = 0
        self.ai_speed_modifier = 0.0
        self.player_lives = 0
        self.round_over = False
        self.round_over_timer = 0
        self.round_winner_text = ""
        self.player_y = 0
        self.ai_y = 0
        self.ball_pos = [0, 0]
        self.ball_vel = [0, 0]
        self.ball_height = 0
        self.ball_height_vel = 0
        self.particles = []

    def _start_new_round(self):
        self.player_lives = 3
        self.round_over = False
        self.round_over_timer = 0
        self.round_winner_text = ""

        self.player_y = self.LOGICAL_HEIGHT / 2
        self.ai_y = self.LOGICAL_HEIGHT / 2

        self.ball_pos = [self.LOGICAL_WIDTH / 2, self.LOGICAL_HEIGHT / 2]
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        if self.np_random.choice([True, False]):
            angle += math.pi

        speed = self.INITIAL_BALL_SPEED
        self.ball_vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.ball_height = 30
        self.ball_height_vel = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_round_wins = 0
        self.ai_round_wins = 0
        self.ai_speed_modifier = 0.0

        self._start_new_round()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        truncated = False
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, truncated, self._get_info()

        if self.round_over:
            self.round_over_timer -= 1
            if self.round_over_timer <= 0:
                if self.player_round_wins >= self.ROUNDS_TO_WIN or self.ai_round_wins >= self.ROUNDS_TO_WIN:
                    self.game_over = True
                else:
                    self._start_new_round()
            
            terminated = self.game_over
            return self._get_observation(), 0, terminated, truncated, self._get_info()

        # 1. Handle player input
        movement = action[0]
        moved = False
        if movement == 1:  # Up
            self.player_y -= self.PLAYER_SPEED
            moved = True
        elif movement == 2:  # Down
            self.player_y += self.PLAYER_SPEED
            moved = True

        if moved:
            reward -= 0.01

        self.player_y = np.clip(self.player_y, self.PADDLE_LOGICAL_HEIGHT / 2, self.LOGICAL_HEIGHT - self.PADDLE_LOGICAL_HEIGHT / 2)

        # 2. Update AI
        ai_speed = self.BASE_AI_SPEED + self.ai_speed_modifier
        target_y = self.ball_pos[1]
        self.ai_y += np.clip(target_y - self.ai_y, -ai_speed, ai_speed)
        self.ai_y = np.clip(self.ai_y, self.PADDLE_LOGICAL_HEIGHT / 2, self.LOGICAL_HEIGHT - self.PADDLE_LOGICAL_HEIGHT / 2)

        # 3. Update Ball
        # 3a. Update position
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        # 3b. Update height (visual only)
        self.ball_height += self.ball_height_vel
        self.ball_height_vel -= 0.5  # Gravity
        if self.ball_height < 0:
            self.ball_height = 0
            self.ball_height_vel *= -0.6

        # 3c. Wall collisions (top/bottom)
        if self.ball_pos[1] <= self.BALL_RADIUS or self.ball_pos[1] >= self.LOGICAL_HEIGHT - self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.LOGICAL_HEIGHT - self.BALL_RADIUS)

        # 3d. Paddle collisions
        # Player
        if self.ball_vel[0] < 0 and self.PLAYER_PADDLE_X < self.ball_pos[0] < self.PLAYER_PADDLE_X + self.PADDLE_THICKNESS:
            if abs(self.player_y - self.ball_pos[1]) < self.PADDLE_LOGICAL_HEIGHT / 2 + self.BALL_RADIUS:
                self.ball_vel[0] *= -1.05  # Speed up
                dy = (self.ball_pos[1] - self.player_y) / (self.PADDLE_LOGICAL_HEIGHT / 2)
                self.ball_vel[1] += dy * 2
                self.ball_vel[1] = np.clip(self.ball_vel[1], -self.INITIAL_BALL_SPEED * 1.5, self.INITIAL_BALL_SPEED * 1.5)
                self.ball_pos[0] = self.PLAYER_PADDLE_X + self.PADDLE_THICKNESS
                self.ball_height_vel = 5  # Pop up
                self._create_particles(self.ball_pos, self.COLOR_PLAYER_ACCENT)
                reward += 0.1

        # AI
        if self.ball_vel[0] > 0 and self.AI_PADDLE_X - self.PADDLE_THICKNESS < self.ball_pos[0] < self.AI_PADDLE_X:
            if abs(self.ai_y - self.ball_pos[1]) < self.PADDLE_LOGICAL_HEIGHT / 2 + self.BALL_RADIUS:
                self.ball_vel[0] *= -1
                dy = (self.ball_pos[1] - self.ai_y) / (self.PADDLE_LOGICAL_HEIGHT / 2)
                self.ball_vel[1] += dy * 2
                self.ball_pos[0] = self.AI_PADDLE_X - self.PADDLE_THICKNESS
                self.ball_height_vel = 4
                self._create_particles(self.ball_pos, self.COLOR_AI_ACCENT)

        # 4. Scoring
        if self.ball_pos[0] < 0:  # AI scores
            self.player_lives -= 1
            if self.player_lives <= 0:
                self.ai_round_wins += 1
                reward -= 1
                self.round_winner_text = "AI WINS ROUND"
                self.round_over = True
                self.round_over_timer = self.FPS * 2
                if self.ai_round_wins >= self.ROUNDS_TO_WIN:
                    reward -= 100
                    self.round_winner_text = "AI WINS THE GAME"
            else:
                self._reset_ball(serve_to_player=True)

        if self.ball_pos[0] > self.LOGICAL_WIDTH:  # Player scores
            self.player_round_wins += 1
            reward += 1
            self.ai_speed_modifier += 0.15
            self.round_winner_text = "PLAYER WINS ROUND"
            self.round_over = True
            self.round_over_timer = self.FPS * 2
            if self.player_round_wins >= self.ROUNDS_TO_WIN:
                reward += 100
                self.round_winner_text = "PLAYER WINS THE GAME"

        # 5. Update particles
        self._update_particles()

        # 6. Check termination
        self.steps += 1
        self.score += reward
        if self.game_over:
            terminated = True
        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        terminated = terminated or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _reset_ball(self, serve_to_player=True):
        self.ball_pos = [self.LOGICAL_WIDTH / 2, self.LOGICAL_HEIGHT / 2]
        angle = self.np_random.uniform(-math.pi / 6, math.pi / 6)
        if not serve_to_player:
            angle += math.pi

        speed = self.INITIAL_BALL_SPEED * 0.8
        self.ball_vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.ball_height = 30
        self.ball_height_vel = 0

    def _to_screen_coords(self, x, y):
        scale_x = 0.9
        scale_y = 0.4
        origin_x = self.WIDTH / 2
        origin_y = 80

        iso_x = (x - y) * scale_x
        iso_y = (x + y) * scale_y

        return int(origin_x + iso_x), int(origin_y + iso_y)

    def _render_game(self):
        # Draw court
        court_points = [
            self._to_screen_coords(0, 0),
            self._to_screen_coords(self.LOGICAL_WIDTH, 0),
            self._to_screen_coords(self.LOGICAL_WIDTH, self.LOGICAL_HEIGHT),
            self._to_screen_coords(0, self.LOGICAL_HEIGHT),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, court_points, self.COLOR_COURT)
        pygame.gfxdraw.aapolygon(self.screen, court_points, self.COLOR_COURT_LINE)

        # Draw center line
        p1 = self._to_screen_coords(self.LOGICAL_WIDTH / 2, 0)
        p2 = self._to_screen_coords(self.LOGICAL_WIDTH / 2, self.LOGICAL_HEIGHT)
        pygame.draw.aaline(self.screen, self.COLOR_COURT_LINE, p1, p2)

        # Draw ball shadow
        shadow_pos = self._to_screen_coords(self.ball_pos[0], self.ball_pos[1])
        shadow_radius = int(self.BALL_RADIUS * (1 - self.ball_height / 100))
        if shadow_radius > 0:
            shadow_surface = pygame.Surface((shadow_radius * 2, shadow_radius * 2), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surface, self.COLOR_SHADOW, shadow_surface.get_rect())
            self.screen.blit(shadow_surface, (shadow_pos[0] - shadow_radius, shadow_pos[1] - shadow_radius))

        # Draw paddles
        self._render_paddle(self.PLAYER_PADDLE_X, self.player_y, self.COLOR_PLAYER, self.COLOR_PLAYER_ACCENT)
        self._render_paddle(self.AI_PADDLE_X, self.ai_y, self.COLOR_AI, self.COLOR_AI_ACCENT)

        # Draw particles
        for p in self.particles:
            pygame.draw.rect(self.screen, p['color'], (*p['pos'], p['size'], p['size']))

        # Draw ball
        ball_screen_pos = self._to_screen_coords(self.ball_pos[0], self.ball_pos[1])
        ball_render_pos = (ball_screen_pos[0], ball_screen_pos[1] - int(self.ball_height))

        # Glow effect for ball
        for i in range(4, 0, -1):
            glow_radius = self.BALL_RADIUS + i * 2
            glow_color = (*self.COLOR_BALL, 50 - i * 10)
            pygame.gfxdraw.filled_circle(self.screen, ball_render_pos[0], ball_render_pos[1], glow_radius, glow_color)

        pygame.gfxdraw.filled_circle(self.screen, ball_render_pos[0], ball_render_pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_render_pos[0], ball_render_pos[1], self.BALL_RADIUS, self.COLOR_BG)

    def _render_paddle(self, logical_x, logical_y, color, accent_color):
        h = self.PADDLE_LOGICAL_HEIGHT / 2
        t = self.PADDLE_THICKNESS
        vh = self.PADDLE_VISUAL_HEIGHT

        # Top face
        top_points = [
            self._to_screen_coords(logical_x - t, logical_y - h),
            self._to_screen_coords(logical_x, logical_y - h),
            self._to_screen_coords(logical_x, logical_y + h),
            self._to_screen_coords(logical_x - t, logical_y + h),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, top_points, accent_color)
        pygame.gfxdraw.aapolygon(self.screen, top_points, color)

        # Front face
        p1 = self._to_screen_coords(logical_x, logical_y - h)
        p2 = self._to_screen_coords(logical_x, logical_y + h)

        front_poly = [
            (p1[0], p1[1]),
            (p2[0], p2[1]),
            (p2[0], p2[1] + vh),
            (p1[0], p1[1] + vh),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, front_poly, color)
        pygame.gfxdraw.aapolygon(self.screen, front_poly, accent_color)

    def _render_ui(self):
        # Round wins display
        player_text = self.font_medium.render(f"P1: {self.player_round_wins}", True, self.COLOR_PLAYER_ACCENT)
        ai_text = self.font_medium.render(f"AI: {self.ai_round_wins}", True, self.COLOR_AI_ACCENT)
        self.screen.blit(player_text, (20, 10))
        self.screen.blit(ai_text, (self.WIDTH - ai_text.get_width() - 20, 10))

        # Player lives display
        life_icon_radius = 8
        for i in range(self.player_lives):
            x = 30 + i * (life_icon_radius * 2 + 5)
            y = 50
            pygame.gfxdraw.filled_circle(self.screen, x, y, life_icon_radius, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, x, y, life_icon_radius, self.COLOR_PLAYER_ACCENT)

        if self.round_over and not self.game_over:
            text_surface = self.font_big.render(self.round_winner_text, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)
        elif self.game_over:
            text_surface = self.font_big.render(self.round_winner_text, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.player_round_wins,
            "steps": self.steps,
            "player_lives": self.player_lives,
            "ai_round_wins": self.ai_round_wins
        }

    def _create_particles(self, pos, color):
        for _ in range(15):
            particle = {
                'pos': list(self._to_screen_coords(pos[0], pos[1])),
                'vel': [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)],
                'lifespan': self.np_random.integers(10, 20),
                'color': color,
                'size': self.np_random.integers(2, 5)
            }
            self.particles.append(particle)

    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in a headless environment
    try:
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            del os.environ["SDL_VIDEODRIVER"]
        
        env = GameEnv(render_mode="rgb_array")
        
        # Use a dummy window to display the game
        pygame.display.set_caption("Pixel Pong")
        real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

        obs, info = env.reset()
        done = False
        
        # Game loop
        while not done:
            # Human input mapping
            movement = 0  # no-op
            action = [0,0,0]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                movement = 2
            
            # Unused actions
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Render the observation to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            real_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(env.FPS)

        env.close()
    except pygame.error as e:
        print(f"Could not run graphical example: {e}")
        print("This is expected in a headless environment.")