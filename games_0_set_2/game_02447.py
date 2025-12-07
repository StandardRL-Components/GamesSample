
# Generated: 2025-08-28T04:50:29.637384
# Source Brief: brief_02447.md
# Brief Index: 2447

        
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
        "Controls: Press Space to use a temporary speed boost. Manage your health carefully!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a snail in a side-view race against time. Strategically use limited boosts to reach the finish line before your health or the clock runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH = 640
        self.HEIGHT = 400
        self.TRACK_LENGTH = 4000
        self.FPS = 30

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
        
        # Fonts
        try:
            self.font_large = pygame.font.Font(None, 72)
            self.font_medium = pygame.font.Font(None, 36)
            self.font_small = pygame.font.Font(None, 24)
        except IOError:
            self.font_large = pygame.font.SysFont("sans", 72)
            self.font_medium = pygame.font.SysFont("sans", 36)
            self.font_small = pygame.font.SysFont("sans", 24)

        # Colors
        self.COLOR_SKY = (135, 206, 235)
        self.COLOR_TRACK = (107, 142, 35)
        self.COLOR_FINISH_1 = (255, 255, 255)
        self.COLOR_FINISH_2 = (30, 30, 30)
        self.COLOR_SNAIL_BODY = (255, 228, 181)
        self.COLOR_SNAIL_SHELL = (139, 69, 19)
        self.COLOR_BOOST = (255, 215, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_HEALTH_FG = (0, 255, 0)
        self.COLOR_HEALTH_BG = (255, 0, 0)

        # Game parameters
        self.TIME_LIMIT_SECONDS = 60.0
        self.MAX_HEALTH = 100
        self.SNAIL_BASE_SPEED = 2.0
        self.SNAIL_BOOST_SPEED = 8.0
        self.BOOST_COST = 20
        self.BOOST_DURATION = 1.0 # seconds
        self.BOOST_COOLDOWN = 1.5 # seconds

        # Parallax background layers
        self.parallax_layers = [
            {"speed": 0.2, "color": (50, 80, 20), "elements": []},
            {"speed": 0.5, "color": (70, 100, 30), "elements": []},
            {"speed": 0.8, "color": (90, 120, 40), "elements": []},
        ]
        self._generate_parallax_elements()

        # Initialize state variables
        self.snail_pos_x = 0.0
        self.snail_speed = 0.0
        self.snail_health = 0.0
        self.time_remaining = 0.0
        self.camera_x = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.boost_active_timer = 0.0
        self.boost_cooldown_timer = 0.0
        self.snail_bob = 0.0

        self.reset()
        self.validate_implementation()
    
    def _generate_parallax_elements(self):
        for layer in self.parallax_layers:
            layer["elements"] = []
            for i in range(100):
                x = random.randint(0, self.TRACK_LENGTH * 2)
                y_base = self.HEIGHT - 100
                h = random.randint(20, 150)
                w = random.randint(10, 40)
                y = y_base - h
                layer["elements"].append(pygame.Rect(x, y, w, h))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.snail_pos_x = 100.0
        self.snail_speed = self.SNAIL_BASE_SPEED
        self.snail_health = self.MAX_HEALTH
        self.time_remaining = self.TIME_LIMIT_SECONDS
        
        self.camera_x = 0.0
        self.particles = []
        self.boost_active_timer = 0.0
        self.boost_cooldown_timer = 0.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        space_pressed = action[1] == 1
        
        # --- Action Logic ---
        if (space_pressed and 
            self.boost_cooldown_timer <= 0 and 
            self.snail_health > self.BOOST_COST):
            # SFX: Boost activate
            self.boost_active_timer = self.BOOST_DURATION
            self.boost_cooldown_timer = self.BOOST_COOLDOWN
            self.snail_health -= self.BOOST_COST
        
        # --- Update Game State ---
        dt = 1 / self.FPS
        self.steps += 1
        self.time_remaining = max(0, self.time_remaining - dt)
        
        if self.boost_active_timer > 0:
            self.boost_active_timer -= dt
            self.snail_speed = self.SNAIL_BOOST_SPEED
            self._create_boost_particles()
        else:
            self.snail_speed = self.SNAIL_BASE_SPEED

        if self.boost_cooldown_timer > 0:
            self.boost_cooldown_timer -= dt

        prev_pos_x = self.snail_pos_x
        self.snail_pos_x += self.snail_speed

        self._update_particles(dt)

        # --- Reward Calculation ---
        reward = 0
        # Reward for forward progress
        reward += (self.snail_pos_x - prev_pos_x) * 0.1
        # Penalty for being idle (encourages boosting)
        if self.snail_speed == self.SNAIL_BASE_SPEED:
            reward -= 0.02
        
        self.score += reward

        # --- Termination Check ---
        terminated = False
        if self.snail_pos_x >= self.TRACK_LENGTH:
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100.0
            self.score += 100.0
        elif self.time_remaining <= 0:
            terminated = True
            self.game_over = True
            reward -= 100.0
            self.score -= 100.0
        elif self.snail_health <= 0:
            # This is an implicit failure, but time is the main loss condition
            # Forcing snail to stop is a better penalty than immediate termination
            self.snail_speed = 0
            self.snail_health = 0

        # Max steps termination
        if self.steps >= 1800: # 60 seconds * 30 FPS
             terminated = True
             self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_boost_particles(self):
        # SFX: Particle fizz
        snail_screen_x = self.WIDTH // 4
        snail_screen_y = self.HEIGHT - 100
        for _ in range(3):
            particle = {
                "pos": [snail_screen_x - 30, snail_screen_y + random.uniform(-10, 10)],
                "vel": [random.uniform(-8, -4), random.uniform(-2, 2)],
                "life": random.uniform(0.5, 1.0),
                "max_life": 1.0,
                "radius": random.uniform(3, 7)
            }
            self.particles.append(particle)
    
    def _update_particles(self, dt):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= dt
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_observation(self):
        # Update camera to follow snail
        self.camera_x = self.snail_pos_x - (self.WIDTH // 4)
        
        # Clear screen
        self.screen.fill(self.COLOR_SKY)
        
        # Render game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Parallax Background
        for layer in self.parallax_layers:
            offset = self.camera_x * layer["speed"]
            for element in layer["elements"]:
                rect = element.copy()
                rect.x -= int(offset)
                if rect.right > 0 and rect.left < self.WIDTH:
                    pygame.draw.rect(self.screen, layer["color"], rect)

        # Track
        track_y = self.HEIGHT - 80
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (0, track_y, self.WIDTH, 80))

        # Finish Line
        finish_x = self.TRACK_LENGTH - self.camera_x
        if finish_x < self.WIDTH + 50:
            check_size = 20
            for i in range(6):
                for j in range(4):
                    color = self.COLOR_FINISH_1 if (i + j) % 2 == 0 else self.COLOR_FINISH_2
                    pygame.draw.rect(self.screen, color, (finish_x + i * check_size, track_y + j * check_size, check_size, check_size))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*self.COLOR_BOOST, alpha)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["radius"]), int(p["pos"][1] - p["radius"])))

        # Snail
        self._render_snail()

    def _render_snail(self):
        snail_screen_x = self.WIDTH // 4
        
        # Animation
        self.snail_bob = math.sin(self.steps * 0.2) * 2
        squash = 1.0
        stretch = 1.0
        if self.boost_active_timer > 0:
            # Squash and stretch effect when boosting
            anim_phase = (self.BOOST_DURATION - self.boost_active_timer) / self.BOOST_DURATION
            squash = 1.0 + math.sin(anim_phase * math.pi) * 0.1
            stretch = 1.0 - math.sin(anim_phase * math.pi) * 0.1
        
        snail_screen_y = self.HEIGHT - 100 + self.snail_bob

        body_w, body_h = int(60 * stretch), int(25 * squash)
        shell_r = int(25 * squash)
        
        # Boost Glow
        if self.boost_active_timer > 0:
            glow_radius = shell_r * 1.5
            glow_alpha = 100
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_BOOST, glow_alpha), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (snail_screen_x - shell_r * 0.5 - glow_radius, snail_screen_y - shell_r - glow_radius))

        # Snail Body
        body_rect = pygame.Rect(snail_screen_x - body_w / 2, snail_screen_y, body_w, body_h)
        pygame.draw.ellipse(self.screen, self.COLOR_SNAIL_BODY, body_rect)
        
        # Snail Shell
        shell_pos = (int(snail_screen_x), int(snail_screen_y - shell_r / 2))
        pygame.draw.circle(self.screen, self.COLOR_SNAIL_SHELL, shell_pos, shell_r)
        
        # Eyes
        eye_r = 3
        eye_y = snail_screen_y + 5
        eye_x1 = snail_screen_x + 15
        eye_x2 = snail_screen_x + 25
        pygame.draw.circle(self.screen, (0,0,0), (int(eye_x1), int(eye_y)), eye_r)
        pygame.draw.circle(self.screen, (0,0,0), (int(eye_x2), int(eye_y)), eye_r)
        
    def _render_ui(self):
        # Timer
        time_text = f"Time: {self.time_remaining:.1f}"
        time_surf = self.font_medium.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))
        
        # Progress Bar
        progress = self.snail_pos_x / self.TRACK_LENGTH
        bar_width = self.WIDTH - 20
        bar_y = self.HEIGHT - 20
        pygame.draw.rect(self.screen, (50,50,50), (10, bar_y, bar_width, 10))
        pygame.draw.rect(self.screen, self.COLOR_BOOST, (10, bar_y, int(bar_width * progress), 10))
        
        # Snail Health Bar
        snail_screen_x = self.WIDTH // 4
        health_percent = self.snail_health / self.MAX_HEALTH
        bar_w, bar_h = 80, 10
        bar_x = snail_screen_x - bar_w / 2
        bar_y = self.HEIGHT - 160
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (bar_x, bar_y, int(bar_w * health_percent), bar_h))

        # Boost Cooldown Indicator
        if self.boost_cooldown_timer > 0:
            cooldown_percent = self.boost_cooldown_timer / self.BOOST_COOLDOWN
            cooldown_text = self.font_small.render("BOOST CD", True, (200, 200, 200))
            self.screen.blit(cooldown_text, (bar_x, bar_y + 12))
            pygame.draw.rect(self.screen, (100, 100, 255), (bar_x, bar_y + 30, bar_w, 5))
            pygame.draw.rect(self.screen, (200, 200, 255), (bar_x, bar_y + 30, int(bar_w * (1-cooldown_percent)), 5))


        # Game Over / Win Text
        if self.game_over:
            text = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            text_surf = self.font_large.render(text, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            # Draw a shadow/outline for readability
            shadow_surf = self.font_large.render(text, True, (0,0,0))
            self.screen.blit(shadow_surf, text_rect.move(3,3))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "health": self.snail_health,
            "progress": self.snail_pos_x / self.TRACK_LENGTH
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Snail Racer")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        space_held = keys[pygame.K_SPACE]
        
        # Convert to MultiDiscrete action
        # Movement (actions[0]) and Shift (actions[2]) are unused in this game
        action = [0, 1 if space_held else 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display
        # The observation is (H, W, C), but pygame needs (W, H, C)
        # and surfarray.make_surface expects it transposed.
        # So we transpose it back.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()