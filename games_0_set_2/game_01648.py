
# Generated: 2025-08-27T17:49:09.684563
# Source Brief: brief_01648.md
# Brief Index: 1648

        
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

    user_guide = (
        "Controls: ↑↓←→ to move the snail. Dodge the red obstacles and collect the blue gems. "
        "Reach the checkered finish line before time runs out!"
    )

    game_description = (
        "Guide a speedy snail through a procedurally generated obstacle course, "
        "collecting gems and racing against the clock to reach the finish line."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.COURSE_LENGTH = 5000
        self.MAX_STEPS = 30 * self.FPS # 30 seconds

        # Colors
        self.COLOR_BG_TOP = (135, 206, 235)
        self.COLOR_BG_BOTTOM = (240, 240, 210)
        self.COLOR_SNAIL_BODY = (153, 255, 153)
        self.COLOR_SNAIL_SHELL = (139, 69, 19)
        self.COLOR_SNAIL_SHELL_HIGHLIGHT = (160, 82, 45)
        self.COLOR_OBSTACLE = (255, 80, 80)
        self.COLOR_OBSTACLE_OUTLINE = (180, 50, 50)
        self.COLOR_GEM = (0, 191, 255)
        self.COLOR_GEM_OUTLINE = (0, 150, 200)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0, 150)
        self.COLOR_FINISH_1 = (50, 50, 50)
        self.COLOR_FINISH_2 = (220, 220, 220)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Arial", 48, bold=True)

        # Pre-render background for performance
        self.background_surface = self._create_background()

        # Initialize state variables
        self.snail_pos = None
        self.camera_x = None
        self.obstacles = None
        self.gems = None
        self.particles = None
        self.base_obstacle_speed = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_message = None

        self.reset()
        self.validate_implementation()

    def _create_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def _generate_level(self):
        self.obstacles.clear()
        self.gems.clear()
        
        current_x = 800
        while current_x < self.COURSE_LENGTH - 400:
            # Place an obstacle chunk
            obstacle_type = self.np_random.choice(['vertical', 'horizontal'])
            y_pos = self.np_random.integers(50, self.HEIGHT - 50)
            
            if obstacle_type == 'vertical':
                obstacle = {
                    "rect": pygame.Rect(current_x, y_pos - 75, 20, 150),
                    "type": "vertical",
                    "center_y": y_pos,
                    "range": self.np_random.integers(50, 150),
                    "phase": self.np_random.random() * 2 * math.pi
                }
            else: # horizontal
                obstacle = {
                    "rect": pygame.Rect(current_x - 75, y_pos - 10, 150, 20),
                    "type": "horizontal",
                    "center_x": current_x,
                    "range": self.np_random.integers(50, 100),
                    "phase": self.np_random.random() * 2 * math.pi
                }
            self.obstacles.append(obstacle)

            # Place some gems around the obstacle
            for _ in range(self.np_random.integers(2, 5)):
                gem_x = current_x + self.np_random.integers(-200, 200)
                gem_y = y_pos + self.np_random.integers(-150, 150)
                if 0 < gem_y < self.HEIGHT - 20:
                    gem_rect = pygame.Rect(gem_x, gem_y, 20, 20)
                    # Ensure gem doesn't spawn inside the obstacle
                    if not gem_rect.colliderect(obstacle['rect']):
                         self.gems.append({
                            "rect": gem_rect,
                            "collected": False,
                            "angle": self.np_random.random() * 360
                        })

            current_x += self.np_random.integers(400, 600)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.SNAIL_SIZE = 30
        self.SNAIL_SPEED = 5
        self.SNAIL_SCREEN_X = 150

        self.snail_pos = pygame.Vector2(self.SNAIL_SCREEN_X, self.HEIGHT / 2)
        self.camera_x = 0
        
        self.obstacles = []
        self.gems = []
        self.particles = []
        
        self.base_obstacle_speed = 1.0
        self._generate_level()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        reward = -0.01  # Penalty for taking a step (encourages speed)

        # 1. Update Snail Position
        prev_x = self.snail_pos.x
        if movement == 1:  # Up
            self.snail_pos.y -= self.SNAIL_SPEED
        elif movement == 2:  # Down
            self.snail_pos.y += self.SNAIL_SPEED
        elif movement == 3:  # Left
            self.snail_pos.x -= self.SNAIL_SPEED
            reward -= 0.02 # Penalty for moving away from goal
        elif movement == 4:  # Right
            self.snail_pos.x += self.SNAIL_SPEED
            reward += 0.1 # Reward for moving towards goal
        
        # Clamp snail to screen bounds
        self.snail_pos.y = np.clip(self.snail_pos.y, self.SNAIL_SIZE / 2, self.HEIGHT - self.SNAIL_SIZE / 2)
        self.snail_pos.x = max(self.camera_x + self.SNAIL_SIZE / 2, self.snail_pos.x)

        # 2. Update Camera
        self.camera_x = self.snail_pos.x - self.SNAIL_SCREEN_X

        # 3. Update Game Elements
        self._update_obstacles()
        self._update_gems()
        self._update_particles()
        
        # 4. Check Collisions & Events
        snail_rect = pygame.Rect(self.snail_pos.x - self.SNAIL_SIZE / 2, self.snail_pos.y - self.SNAIL_SIZE / 2, self.SNAIL_SIZE, self.SNAIL_SIZE)
        
        # Obstacle collision
        for obs in self.obstacles:
            if snail_rect.colliderect(obs["rect"]):
                # sfx: player_hit
                self.game_over = True
                self.win_message = "CRASHED!"
                reward = -100
                break
        
        # Gem collection
        if not self.game_over:
            for gem in self.gems:
                if not gem["collected"] and snail_rect.colliderect(gem["rect"]):
                    # sfx: gem_collect
                    gem["collected"] = True
                    self.score += 1
                    reward += 1.0
                    self._create_particles(gem["rect"].center)

        # 5. Check Win/Loss Conditions
        terminated = self.game_over
        if not terminated:
            if self.snail_pos.x >= self.COURSE_LENGTH:
                # sfx: level_complete
                self.game_over = True
                terminated = True
                self.win_message = "FINISH!"
                reward = 100
            elif self.steps >= self.MAX_STEPS:
                self.game_over = True
                terminated = True
                self.win_message = "TIME'S UP!"
                reward -= 10 # Small penalty for timeout
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_obstacles(self):
        current_speed_multiplier = self.base_obstacle_speed + (self.steps // 100) * 0.02
        for obs in self.obstacles:
            phase = obs["phase"] + self.steps * 0.05 * current_speed_multiplier
            if obs["type"] == 'vertical':
                obs["rect"].centery = obs["center_y"] + math.sin(phase) * obs["range"]
            else: # horizontal
                obs["rect"].centerx = obs["center_x"] + math.sin(phase) * obs["range"]

    def _update_gems(self):
        for gem in self.gems:
            gem["angle"] = (gem["angle"] + 5) % 360
            
    def _create_particles(self, pos):
        for _ in range(15):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifespan": self.np_random.integers(15, 30),
                "color": self.COLOR_GEM
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # friction
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.blit(self.background_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw finish line
        finish_x_screen = self.COURSE_LENGTH - self.camera_x
        if finish_x_screen < self.WIDTH:
            check_size = 20
            for y in range(0, self.HEIGHT, check_size):
                for x in range(int(finish_x_screen), int(finish_x_screen + 40), check_size):
                    row = y // check_size
                    col = (x - int(finish_x_screen)) // check_size
                    color = self.COLOR_FINISH_1 if (row + col) % 2 == 0 else self.COLOR_FINISH_2
                    pygame.draw.rect(self.screen, color, (x, y, check_size, check_size))

        # Draw obstacles
        for obs in self.obstacles:
            screen_rect = obs["rect"].move(-self.camera_x, 0)
            if self.screen.get_rect().colliderect(screen_rect):
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, screen_rect, border_radius=5)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect.inflate(-4, -4), border_radius=5)

        # Draw gems
        for gem in self.gems:
            if not gem["collected"]:
                screen_rect = gem["rect"].move(-self.camera_x, 0)
                if self.screen.get_rect().colliderect(screen_rect):
                    self._draw_rotated_square(self.screen, gem["rect"].centerx - self.camera_x, gem["rect"].centery, gem["angle"], gem["rect"].width * 0.7)

        # Draw particles
        for p in self.particles:
            screen_pos = p["pos"] - pygame.Vector2(self.camera_x, 0)
            size = max(1, p["lifespan"] / 5)
            pygame.draw.circle(self.screen, p["color"], (int(screen_pos.x), int(screen_pos.y)), int(size))
            
        # Draw snail
        self._draw_snail(self.screen)

    def _draw_snail(self, surface):
        screen_pos = (self.SNAIL_SCREEN_X, self.snail_pos.y)
        
        # Body
        body_rect = pygame.Rect(0, 0, self.SNAIL_SIZE * 1.2, self.SNAIL_SIZE * 0.8)
        body_rect.center = screen_pos
        pygame.gfxdraw.filled_ellipse(surface, int(body_rect.centerx), int(body_rect.centery), int(body_rect.width/2), int(body_rect.height/2), self.COLOR_SNAIL_BODY)
        pygame.gfxdraw.aaellipse(surface, int(body_rect.centerx), int(body_rect.centery), int(body_rect.width/2), int(body_rect.height/2), self.COLOR_SNAIL_BODY)
        
        # Shell
        shell_bob = math.sin(self.steps * 0.2) * 2
        shell_rect = pygame.Rect(0, 0, self.SNAIL_SIZE, self.SNAIL_SIZE)
        shell_rect.center = (screen_pos[0] + 5, screen_pos[1] - 10 + shell_bob)
        pygame.gfxdraw.filled_ellipse(surface, int(shell_rect.centerx), int(shell_rect.centery), int(shell_rect.width/2), int(shell_rect.height/2), self.COLOR_SNAIL_SHELL)
        pygame.gfxdraw.arc(surface, int(shell_rect.centerx), int(shell_rect.centery), int(shell_rect.width/2 - 2), 180, 360, self.COLOR_SNAIL_SHELL_HIGHLIGHT)
        pygame.gfxdraw.aaellipse(surface, int(shell_rect.centerx), int(shell_rect.centery), int(shell_rect.width/2), int(shell_rect.height/2), self.COLOR_SNAIL_SHELL_HIGHLIGHT)

        # Eyes
        eye_pos1 = (screen_pos[0] - 10, screen_pos[1] - 5)
        eye_pos2 = (screen_pos[0] - 5, screen_pos[1] - 5)
        pygame.draw.circle(surface, (255, 255, 255), eye_pos1, 3)
        pygame.draw.circle(surface, (255, 255, 255), eye_pos2, 3)
        pygame.draw.circle(surface, (0, 0, 0), eye_pos1, 1)
        pygame.draw.circle(surface, (0, 0, 0), eye_pos2, 1)

    def _draw_rotated_square(self, surface, x, y, angle, size):
        points = []
        for i in range(4):
            px = x + size * math.cos(math.radians(angle + 45 + 90 * i))
            py = y + size * math.sin(math.radians(angle + 45 + 90 * i))
            points.append((int(px), int(py)))
        pygame.gfxdraw.aapolygon(surface, points, self.COLOR_GEM_OUTLINE)
        pygame.gfxdraw.filled_polygon(surface, points, self.COLOR_GEM)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        self._draw_text_shadowed(self.screen, score_text, (10, 10), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Time
        remaining_time = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"TIME: {remaining_time:.1f}"
        self._draw_text_shadowed(self.screen, time_text, (10, 40), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Progress bar
        progress = self.snail_pos.x / self.COURSE_LENGTH
        bar_width = self.WIDTH - 20
        bar_height = 10
        pygame.draw.rect(self.screen, self.COLOR_TEXT_SHADOW, (10, self.HEIGHT - 20, bar_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_SNAIL_BODY, (10, self.HEIGHT - 20, bar_width * progress, bar_height), border_radius=3)

        if self.game_over:
            self._draw_text_shadowed(self.screen, self.win_message, self.screen.get_rect().center, self.font_game_over, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, center=True)

    def _draw_text_shadowed(self, surface, text, pos, font, color, shadow_color, center=False):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        shadow_surf = font.render(text, True, shadow_color)
        shadow_rect = shadow_surf.get_rect()

        if center:
            text_rect.center = pos
            shadow_rect.center = (pos[0] + 2, pos[1] + 2)
        else:
            text_rect.topleft = pos
            shadow_rect.topleft = (pos[0] + 2, pos[1] + 2)
        
        surface.blit(shadow_surf, shadow_rect)
        surface.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "progress": self.snail_pos.x / self.COURSE_LENGTH
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a display for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Speedy Snail")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print(env.user_guide)

    while not done:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        # Pygame uses (width, height), numpy uses (height, width), so we need to transpose.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()