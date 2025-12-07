
# Generated: 2025-08-27T13:21:12.207650
# Source Brief: brief_00340.md
# Brief Index: 340

        
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
        "Controls: Use ↑←→ to jump from a platform. Hold Shift for a high-power jump. Press ↓ to fall faster."
    )

    game_description = (
        "Hop through a dangerous asteroid field, collect stars, and manage your fuel to survive."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000
        
        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_THRUST = (255, 255, 100)
        self.COLOR_PLATFORM = (100, 110, 130)
        self.COLOR_STAR = (255, 220, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_FUEL_GREEN = (40, 200, 80)
        self.COLOR_FUEL_YELLOW = (250, 200, 50)
        self.COLOR_FUEL_RED = (220, 50, 50)

        # Physics
        self.GRAVITY = 0.4
        self.JUMP_POWER_LOW = 9.0
        self.JUMP_POWER_HIGH = 12.0
        self.JUMP_ANGLE_MULTIPLIER = 0.75
        self.FAST_FALL_ACCEL = 0.8
        self.MAX_FALL_SPEED = 15.0

        # Game Mechanics
        self.MAX_FUEL = 100.0
        self.BASE_JUMP_FUEL_COST = 5.0
        self.NUM_STARS = 5

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 50)

        # Game State (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 0
        self.fuel = 0.0
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.platforms = []
        self.stars = []
        self.particles = []
        self.star_twinkle_angle = 0
        self.background_stars = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options and "level" in options:
            self.level = options["level"]
        else:
            self.level = 1

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fuel = self.MAX_FUEL
        
        self._generate_level()

        start_platform = self.platforms[0]
        self.player_pos = pygame.Vector2(start_platform.centerx, start_platform.top - 15)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = True
        
        self.particles = []
        self.star_twinkle_angle = 0
        
        if not self.background_stars:
            for _ in range(150):
                self.background_stars.append(
                    (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 3))
                )

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Small penalty for existing
        prev_y = self.player_pos.y

        # --- Handle Input ---
        if self.on_ground and movement != 0 and self.fuel > 0:
            self.on_ground = False
            power = self.JUMP_POWER_HIGH if shift_held else self.JUMP_POWER_LOW
            fuel_cost = self.BASE_JUMP_FUEL_COST + (self.level - 1) * 0.05
            self.fuel = max(0, self.fuel - fuel_cost)

            if movement == 1:  # Up
                self.player_vel.y = -power
                self.player_vel.x = 0
            elif movement == 3:  # Left
                self.player_vel.y = -power * self.JUMP_ANGLE_MULTIPLIER
                self.player_vel.x = -power * self.JUMP_ANGLE_MULTIPLIER
            elif movement == 4:  # Right
                self.player_vel.y = -power * self.JUMP_ANGLE_MULTIPLIER
                self.player_vel.x = power * self.JUMP_ANGLE_MULTIPLIER
            
            # // sound: jump.wav
            self._create_particles(self.player_pos, 15, self.COLOR_PLAYER_THRUST)

        elif not self.on_ground and movement == 2:  # Down (fast fall)
            self.player_vel.y = min(self.player_vel.y + self.FAST_FALL_ACCEL, self.MAX_FALL_SPEED)

        # --- Update Physics ---
        self.player_vel.y += self.GRAVITY
        self.player_pos += self.player_vel
        self.on_ground = False # Assume not on ground until collision check
        
        # --- Collisions ---
        player_rect = self._get_player_rect()

        # Platform collision
        if self.player_vel.y > 0:
            for plat in self.platforms:
                if player_rect.colliderect(plat) and player_rect.bottom < plat.bottom:
                    self.player_pos.y = plat.top - player_rect.height / 2
                    self.player_vel.y = 0
                    self.player_vel.x *= 0.5 # friction
                    self.on_ground = True
                    # // sound: land.wav
                    break
        
        # Star collection
        collected_star_indices = []
        for i, star_pos in enumerate(self.stars):
            if player_rect.collidepoint(star_pos):
                collected_star_indices.append(i)
                reward += 10
                self.score += 10
                # // sound: star_collect.wav
                self._create_particles(star_pos, 20, self.COLOR_STAR)
        
        if collected_star_indices:
            self.stars = [s for i, s in enumerate(self.stars) if i not in collected_star_indices]

        # --- Update State ---
        self.steps += 1
        self.star_twinkle_angle = (self.star_twinkle_angle + 0.1) % (2 * math.pi)

        # Vertical movement reward
        if self.player_pos.y < prev_y:
            reward += 0.1 # Reward for moving up
        elif self.player_pos.y > prev_y:
            reward -= 0.1 # Penalty for moving down
        
        # --- Termination ---
        terminated = False
        win = len(self.stars) == 0
        
        if win:
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
        elif self.player_pos.y > self.HEIGHT + 50:
            reward -= 50
            self.score -= 50
            terminated = True
            self.game_over = True
        elif self.fuel <= 0 and not self.on_ground:
            reward -= 50
            self.score -= 50
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        self.score = max(self.score, -50) # Clamp min score

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_level(self):
        self.platforms = []
        self.stars = []
        
        # Starting platform
        start_plat = pygame.Rect(self.WIDTH // 2 - 50, self.HEIGHT - 80, 100, 20)
        self.platforms.append(start_plat)
        last_plat = start_plat
        
        # Procedural platforms
        min_gap_x = 50 + self.level * 5
        max_gap_x = 150 + self.level * 5
        min_gap_y = -80
        max_gap_y = 50

        for _ in range(15):
            width = random.randint(60, 120)
            dx = random.randint(min_gap_x, max_gap_x) * random.choice([-1, 1])
            dy = random.randint(min_gap_y, max_gap_y)
            
            new_x = last_plat.centerx + dx
            new_y = last_plat.y + dy
            
            # Clamp to screen
            new_x = np.clip(new_x, width // 2, self.WIDTH - width // 2)
            new_y = np.clip(new_y, 50, self.HEIGHT - 50)
            
            new_plat = pygame.Rect(0, 0, width, 20)
            new_plat.center = (new_x, new_y)
            
            # Ensure no overlap
            if not any(new_plat.colliderect(p) for p in self.platforms):
                self.platforms.append(new_plat)
                last_plat = new_plat

        # Procedural stars
        while len(self.stars) < self.NUM_STARS:
            plat = random.choice(self.platforms[1:]) # Don't spawn on first platform
            star_pos = (plat.centerx + random.randint(-30, 30), plat.top - 30 + random.randint(-20, 20))
            
            # Ensure star is not inside a platform
            valid = True
            for p in self.platforms:
                if p.collidepoint(star_pos):
                    valid = False
                    break
            if valid:
                self.stars.append(star_pos)


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fuel": self.fuel,
            "level": self.level,
            "stars_remaining": len(self.stars),
        }

    def _get_player_rect(self):
        return pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 15, 20, 20)

    def _render_game(self):
        # Background stars
        for x, y, size in self.background_stars:
            color_val = 50 + size * 20
            pygame.draw.rect(self.screen, (color_val, color_val, color_val), (x, y, size, size))
            
        # Particles
        self._update_and_draw_particles()
        
        # Platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in self.COLOR_PLATFORM), plat.inflate(-4,-4), border_radius=3)

        # Stars
        twinkle_scale = 1.0 + 0.1 * math.sin(self.star_twinkle_angle * 5)
        for x, y in self.stars:
            self._draw_star(self.screen, self.COLOR_STAR, (x, y), 12 * twinkle_scale)
            
        # Player
        player_rect = self._get_player_rect()
        points = [
            (player_rect.centerx, player_rect.top),
            (player_rect.right, player_rect.bottom),
            (player_rect.left, player_rect.bottom)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Player thrust
        if not self.on_ground and self.player_vel.y < 0:
            thrust_size = 5 + min(abs(self.player_vel.y), 5)
            thrust_points = [
                (player_rect.centerx - 4, player_rect.bottom),
                (player_rect.centerx + 4, player_rect.bottom),
                (player_rect.centerx, player_rect.bottom + thrust_size)
            ]
            pygame.gfxdraw.aapolygon(self.screen, thrust_points, self.COLOR_PLAYER_THRUST)
            pygame.gfxdraw.filled_polygon(self.screen, thrust_points, self.COLOR_PLAYER_THRUST)
    
    def _render_ui(self):
        # Star counter
        self._draw_star(self.screen, self.COLOR_STAR, (30, 30), 12)
        star_text = self.font_ui.render(f"{self.NUM_STARS - len(self.stars)} / {self.NUM_STARS}", True, self.COLOR_TEXT)
        self.screen.blit(star_text, (50, 20))
        
        # Fuel gauge
        fuel_percent = self.fuel / self.MAX_FUEL
        bar_width = 150
        bar_height = 18
        
        fuel_color = self.COLOR_FUEL_GREEN
        if fuel_percent < 0.5:
            fuel_color = self.COLOR_FUEL_YELLOW
        if fuel_percent < 0.2:
            fuel_color = self.COLOR_FUEL_RED
            
        fuel_bar_rect = pygame.Rect(self.WIDTH - bar_width - 20, 20, bar_width * fuel_percent, bar_height)
        border_rect = pygame.Rect(self.WIDTH - bar_width - 20, 20, bar_width, bar_height)
        
        pygame.draw.rect(self.screen, fuel_color, fuel_bar_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, border_rect, 2, border_radius=4)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        
        win = len(self.stars) == 0
        text = "LEVEL COMPLETE" if win else "GAME OVER"
        color = self.COLOR_STAR if win else self.COLOR_FUEL_RED
        
        text_surf = self.font_big.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        overlay.blit(text_surf, text_rect)
        
        self.screen.blit(overlay, (0, 0))

    def _draw_star(self, surface, color, center, size):
        points = []
        for i in range(10):
            angle = math.radians(i * 36) + self.star_twinkle_angle
            radius = size if i % 2 == 0 else size * 0.5
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            velocity = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': velocity,
                'life': random.randint(15, 30),
                'color': color
            })

    def _update_and_draw_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
            size = max(0, p['life'] / 10)
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], p['pos'], size)
        self.particles = [p for p in self.particles if p['life'] > 0]
        
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Hopping Spaceship")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Game Step ---
        # NOTE: For human play, we use a different action mapping.
        # The brief required a jump-on-platform mechanic.
        # Let's map spacebar to the jump action for better playability.
        # The AI will learn to use the movement keys to jump.
        
        human_movement = 0
        if keys[pygame.K_SPACE]:
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                human_movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                human_movement = 4
            else:
                human_movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
             human_movement = 2

        action = [human_movement, 0, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                total_reward = 0
                obs, info = env.reset()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}")
            total_reward = 0
            obs, info = env.reset()

        clock.tick(env.FPS)
        
    pygame.quit()