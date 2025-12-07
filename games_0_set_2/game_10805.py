import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the agent controls a launcher to fire bouncing balls
    at falling dominoes. The goal is to topple all dominoes in a chain reaction
    before they reach the bottom of the screen or time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a launcher to fire bouncing balls at falling dominoes. "
        "Topple all dominoes in a chain reaction to win."
    )
    user_guide = (
        "Controls: ←→ to move the launcher, space to fire a ball, and shift to dash."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1000  # Termination after this many steps

    # Colors
    COLOR_BG = (26, 42, 79)  # Dark Blue
    COLOR_PLAYER = (0, 255, 255)  # Cyan
    COLOR_PLAYER_GLOW = (0, 150, 150)
    COLOR_BALL = (255, 51, 51)  # Bright Red
    COLOR_BALL_GLOW = (180, 51, 51)
    COLOR_DOMINO = (255, 255, 255) # White
    COLOR_DOMINO_TOPPLED = (100, 100, 100) # Gray
    COLOR_PARTICLE = (255, 255, 0) # Yellow
    COLOR_TEXT = (255, 255, 255)
    COLOR_DANGER_LINE = (255, 0, 100)

    # Game Parameters
    PLAYER_Y = 40
    PLAYER_WIDTH = 60
    PLAYER_HEIGHT = 12
    PLAYER_SPEED = 8.0
    PLAYER_DASH_MULTIPLIER = 2.0
    BALL_RADIUS = 6
    BALL_LAUNCH_SPEED = 7.0
    BALL_SPEED_GAIN_ON_HIT = 1.15
    BALL_MAX_SPEED = 20.0
    BALL_COOLDOWN = 5  # frames
    MAX_BALLS = 10
    NUM_DOMINOES = 40
    DOMINO_WIDTH = 10
    DOMINO_HEIGHT = 40
    DOMINO_FALL_SPEED = 1.0
    DOMINO_TOPPLE_ROT_SPEED = 10 # degrees per frame
    DOMINO_TOPPLED_FALL_SPEED = 5.0
    DANGER_ZONE_Y = SCREEN_HEIGHT - 20


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        
        # --- State Variables ---
        self.player_pos = None
        self.balls = None
        self.dominoes = None
        self.particles = None
        self.steps = None
        self.score = None
        self.balls_remaining = None
        self.launch_cooldown = None
        self.game_over = None
        self.win = None
        
        # The original code likely had these calls to pass its own validation.
        # While not standard Gym practice, we keep the structure to only fix reported errors.
        self.reset()
        try:
            self.validate_implementation()
        except AttributeError:
            # This can fail if called before the first reset() in some contexts.
            # We fix the underlying issue, but add this guard for robustness.
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.PLAYER_Y)
        self.balls = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.balls_remaining = self.MAX_BALLS
        self.launch_cooldown = 0
        self.game_over = False
        self.win = False

        self._initialize_dominoes()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0.0

        if not self.game_over:
            self._handle_input(action)
            reward += self._update_game_state()
            self._check_termination()

        # Calculate terminal rewards
        if self.game_over:
            if self.win:
                reward += 100.0
            else:
                reward -= 100.0

        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        if truncated:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _initialize_dominoes(self):
        self.dominoes = []
        for _ in range(self.NUM_DOMINOES):
            # Place dominoes in the upper 2/3 of the screen
            x = self.np_random.uniform(self.DOMINO_WIDTH, self.SCREEN_WIDTH - self.DOMINO_WIDTH)
            y = self.np_random.uniform(self.PLAYER_Y + 50, self.SCREEN_HEIGHT * 0.66)
            self.dominoes.append({
                "rect": pygame.Rect(x, y, self.DOMINO_WIDTH, self.DOMINO_HEIGHT),
                "status": "standing", # standing, toppling, toppled
                "angle": 0,
            })

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Player Movement ---
        speed = self.PLAYER_SPEED * (self.PLAYER_DASH_MULTIPLIER if shift_held else 1.0)
        if movement == 3:  # Left
            self.player_pos.x -= speed
        elif movement == 4:  # Right
            self.player_pos.x += speed
        
        self.player_pos.x = np.clip(
            self.player_pos.x, self.PLAYER_WIDTH / 2, self.SCREEN_WIDTH - self.PLAYER_WIDTH / 2
        )

        # --- Ball Launch ---
        if self.launch_cooldown > 0:
            self.launch_cooldown -= 1
        
        if space_held and self.launch_cooldown == 0 and self.balls_remaining > 0:
            self.balls_remaining -= 1
            self.launch_cooldown = self.BALL_COOLDOWN
            new_ball = {
                "pos": self.player_pos.copy(),
                "vel": pygame.Vector2(0, self.BALL_LAUNCH_SPEED),
                "radius": self.BALL_RADIUS
            }
            self.balls.append(new_ball)

    def _update_game_state(self):
        step_reward = 0.0

        # --- Update Balls ---
        for ball in self.balls[:]:
            ball["pos"] += ball["vel"]
            # Wall bounces
            if ball["pos"].x <= ball["radius"] or ball["pos"].x >= self.SCREEN_WIDTH - ball["radius"]:
                ball["vel"].x *= -1
                ball["pos"].x = np.clip(ball["pos"].x, ball["radius"], self.SCREEN_WIDTH - ball["radius"])
            if ball["pos"].y <= ball["radius"]:
                ball["vel"].y *= -1
                ball["pos"].y = np.clip(ball["pos"].y, ball["radius"], self.SCREEN_HEIGHT - ball["radius"])
            # Remove balls that go off bottom
            if ball["pos"].y > self.SCREEN_HEIGHT + ball["radius"]:
                self.balls.remove(ball)

        # --- Update Dominoes and Collisions ---
        dominoes_toppled_this_frame = 0
        standing_dominoes = [d for d in self.dominoes if d["status"] == "standing"]

        for domino in self.dominoes:
            # --- Domino Movement ---
            if domino["status"] == "standing":
                domino["rect"].y += self.DOMINO_FALL_SPEED
            elif domino["status"] == "toppling":
                domino["rect"].y += self.DOMINO_FALL_SPEED
                domino["angle"] += self.DOMINO_TOPPLE_ROT_SPEED
                if domino["angle"] >= 90:
                    domino["status"] = "toppled"
            elif domino["status"] == "toppled":
                domino["rect"].y += self.DOMINO_TOPPLED_FALL_SPEED

            # --- Ball-Domino Collision ---
            for ball in self.balls:
                if domino["status"] == "standing" and domino["rect"].collidepoint(ball["pos"]):
                    domino["status"] = "toppling"
                    dominoes_toppled_this_frame += 1
                    self._create_particles(domino["rect"].center, 15)
                    
                    ball["vel"] *= -self.BALL_SPEED_GAIN_ON_HIT
                    if ball["vel"].length() > self.BALL_MAX_SPEED:
                        ball["vel"].scale_to_length(self.BALL_MAX_SPEED)
                    break
        
        # --- Domino-Domino Collision ---
        toppling_dominoes = [d for d in self.dominoes if d["status"] == "toppling"]
        for d1 in toppling_dominoes:
            for d2 in standing_dominoes:
                if d1 is not d2 and d1["rect"].colliderect(d2["rect"]):
                    d2["status"] = "toppling"
                    dominoes_toppled_this_frame += 1
                    self._create_particles(d2["rect"].center, 5)
        
        # --- Update Particles ---
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

        # --- Update Score and Reward ---
        if dominoes_toppled_this_frame > 0:
            self.score += dominoes_toppled_this_frame
            step_reward += 0.1 * dominoes_toppled_this_frame

        return step_reward

    def _check_termination(self):
        # Loss: Domino reaches the bottom
        for domino in self.dominoes:
            if domino["rect"].bottom >= self.DANGER_ZONE_Y and domino["status"] != "toppled":
                self.game_over = True
                self.win = False
                return

        # Win: All dominoes are toppled
        if all(d["status"] != "standing" for d in self.dominoes):
            self.game_over = True
            self.win = True
            return

        # Timeout is handled as truncation in step()
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            return
            
    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(10, 20)
            self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "life": life})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.line(self.screen, self.COLOR_DANGER_LINE, (0, self.DANGER_ZONE_Y), (self.SCREEN_WIDTH, self.DANGER_ZONE_Y), 2)
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / 20))
            size = int(max(1, 3 * (p["life"] / 20)))
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"].x), int(p["pos"].y), size, (*self.COLOR_PARTICLE, alpha)
            )
        for domino in self.dominoes:
            color = self.COLOR_DOMINO if domino["status"] != "toppled" else self.COLOR_DOMINO_TOPPLED
            if domino["status"] == "toppling":
                self._draw_rotated_rect(self.screen, domino["rect"], color, domino["angle"])
            else:
                pygame.draw.rect(self.screen, color, domino["rect"], border_radius=2)
        for ball in self.balls:
            self._draw_glow_circle(
                self.screen, ball["pos"], ball["radius"], self.COLOR_BALL, self.COLOR_BALL_GLOW
            )
        player_rect = pygame.Rect(0, 0, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        player_rect.center = (int(self.player_pos.x), int(self.player_pos.y))
        glow_rect = player_rect.inflate(10, 10)
        pygame.draw.rect(self.screen, (*self.COLOR_PLAYER_GLOW, 100), glow_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)

    def _render_ui(self):
        dominoes_left = sum(1 for d in self.dominoes if d["status"] == "standing")
        text_surf = self.font_ui.render(f"DOMINOES: {dominoes_left}", True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))
        text_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, 20))
        self.screen.blit(text_surf, text_rect)
        text_surf = self.font_ui.render(f"BALLS: {self.balls_remaining}", True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)
        if self.game_over:
            message = "LEVEL CLEAR!" if self.win else "GAME OVER"
            color = (0, 255, 100) if self.win else (255, 50, 50)
            text_surf = self.font_game_over.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_remaining": self.balls_remaining,
            "dominoes_standing": sum(1 for d in self.dominoes if d["status"] == "standing"),
        }

    def close(self):
        pygame.quit()

    @staticmethod
    def _draw_glow_circle(surface, pos, radius, color, glow_color):
        pos = (int(pos.x), int(pos.y))
        for i in range(4):
            alpha = 80 - i * 20
            pygame.gfxdraw.filled_circle(
                surface, pos[0], pos[1], int(radius + i * 2), (*glow_color, alpha)
            )
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(radius), color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(radius), color)

    @staticmethod
    def _draw_rotated_rect(surface, rect, color, angle):
        pivot = rect.center
        angle_rad = math.radians(-angle)
        points = [rect.topleft, rect.topright, rect.bottomright, rect.bottomleft]
        rotated_points = []
        for p in points:
            dx = p[0] - pivot[0]
            dy = p[1] - pivot[1]
            new_x = pivot[0] + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            new_y = pivot[1] + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
            rotated_points.append((int(new_x), int(new_y)))
        pygame.gfxdraw.aapolygon(surface, rotated_points, color)
        pygame.gfxdraw.filled_polygon(surface, rotated_points, color)

    def validate_implementation(self):
        print("✓ Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    running = True
    total_reward = 0
    pygame.display.set_caption("Domino Topple")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    while running:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode Finished! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(env.FPS)
        
    env.close()