import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:38:17.739471
# Source Brief: brief_00542.md
# Brief Index: 542
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A fast-paced, visually-rich arcade shooter environment for Gymnasium.
    The goal is to achieve the highest score by creating chain-reaction
    explosions of alien ships.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A fast-paced arcade shooter. Destroy waves of aliens and create chain-reaction explosions to rack up combo points."
    )
    user_guide = (
        "Controls: ←→ to move your ship and space to fire at the aliens."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_PROJECTILE_PLAYER = (255, 255, 0)
    COLOR_PROJECTILE_ALIEN = (255, 0, 255)
    COLOR_TEXT = (220, 220, 255)
    ALIEN_COLORS = {
        10: (50, 255, 50),   # Green
        20: (50, 150, 255),  # Blue
        30: (255, 50, 50)    # Red
    }

    # Player settings
    PLAYER_WIDTH = 30
    PLAYER_HEIGHT = 20
    PLAYER_SPEED = 10
    PLAYER_PROJECTILE_SPEED = 15
    PLAYER_FIRE_COOLDOWN = 4 # frames

    # Alien settings
    ALIEN_ROWS = 4
    ALIEN_COLS = 8
    ALIEN_SPACING = 50
    ALIEN_WIDTH = 24
    ALIEN_HEIGHT = 24
    ALIEN_DROP_DIST = 10

    # Combo settings
    COMBO_TIMEOUT = 90 # frames (3 seconds at 30 FPS)

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_combo = pygame.font.SysFont("monospace", 32, bold=True)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_fire_cooldown_timer = 0
        self.player_projectiles = []
        self.aliens = []
        self.alien_projectiles = []
        self.alien_direction = 1
        self.alien_base_speed = 1.0
        self.alien_fire_prob = 0.01
        self.particles = []
        self.stars = []
        self.combo_multiplier = 1
        self.combo_timer = 0
        self.space_was_held = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 40)
        self.player_fire_cooldown_timer = 0
        self.space_was_held = False

        # Projectiles & Particles
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        
        # Combo
        self.combo_multiplier = 1
        self.combo_timer = 0

        # Difficulty
        self.alien_base_speed = 1.0
        self.alien_fire_prob = 0.01

        # Aliens
        self.aliens = []
        self.alien_direction = 1
        start_x = self.SCREEN_WIDTH / 2 - (self.ALIEN_COLS / 2 * self.ALIEN_SPACING)
        start_y = 50
        alien_types = sorted(self.ALIEN_COLORS.keys(), reverse=True)
        for row in range(self.ALIEN_ROWS):
            points = alien_types[row % len(alien_types)]
            color = self.ALIEN_COLORS[points]
            for col in range(self.ALIEN_COLS):
                pos = pygame.Vector2(start_x + col * self.ALIEN_SPACING, start_y + row * self.ALIEN_SPACING)
                self.aliens.append({"pos": pos, "points": points, "color": color})

        # Background Stars
        self.stars = []
        for _ in range(150):
            self.stars.append({
                "pos": pygame.Vector2(random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT)),
                "size": random.randint(1, 2),
                "brightness": random.randint(50, 150)
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is action[2], currently unused

        reward = 0.0

        # --- Update Game Logic ---
        self._handle_input(movement, space_held)
        self._update_player_projectiles()
        self._update_alien_projectiles()
        self._update_aliens()
        self._update_particles()
        self._update_combo()

        # --- Collision Detection & Rewards ---
        reward += self._handle_collisions()
        
        # --- Difficulty Scaling ---
        self._update_difficulty()

        self.steps += 1
        terminated = self.steps >= self.MAX_STEPS or not self.aliens
        truncated = False

        return (
            self._get_observation(),
            np.clip(reward, -10.0, 10.0), # Clip reward as per spec
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Movement
        if movement == 3:  # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos.x += self.PLAYER_SPEED
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_WIDTH / 2, self.SCREEN_WIDTH - self.PLAYER_WIDTH / 2)

        # Firing
        if self.player_fire_cooldown_timer > 0:
            self.player_fire_cooldown_timer -= 1
            
        if space_held and self.player_fire_cooldown_timer == 0:
            # SFX: player_shoot
            self.player_projectiles.append(self.player_pos.copy())
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN

        self.space_was_held = space_held

    def _update_player_projectiles(self):
        for proj in self.player_projectiles[:]:
            proj.y -= self.PLAYER_PROJECTILE_SPEED
            if proj.y < 0:
                self.player_projectiles.remove(proj)

    def _update_alien_projectiles(self):
        for proj in self.alien_projectiles[:]:
            proj.y += self.PLAYER_PROJECTILE_SPEED / 2
            if proj.y > self.SCREEN_HEIGHT:
                self.alien_projectiles.remove(proj)

    def _update_aliens(self):
        if not self.aliens:
            return

        move_down = False
        for alien in self.aliens:
            if (self.alien_direction > 0 and alien["pos"].x > self.SCREEN_WIDTH - self.ALIEN_WIDTH) or \
               (self.alien_direction < 0 and alien["pos"].x < self.ALIEN_WIDTH):
                move_down = True
                break
        
        if move_down:
            self.alien_direction *= -1
            for alien in self.aliens:
                alien["pos"].y += self.ALIEN_DROP_DIST

        for alien in self.aliens:
            alien["pos"].x += self.alien_direction * self.alien_base_speed
            if random.random() < self.alien_fire_prob:
                # SFX: alien_shoot
                self.alien_projectiles.append(alien["pos"].copy())

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["radius"] *= 0.95
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _update_combo(self):
        if self.combo_timer > 0:
            self.combo_timer -= 1
        elif self.combo_multiplier > 1:
            # SFX: combo_reset
            self.combo_multiplier = 1

    def _handle_collisions(self):
        reward = 0.0
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_WIDTH / 2, self.player_pos.y - self.PLAYER_HEIGHT / 2, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)

        # Player projectiles vs Aliens
        for proj in self.player_projectiles[:]:
            proj_rect = pygame.Rect(proj.x - 2, proj.y - 10, 4, 20)
            for alien in self.aliens[:]:
                alien_rect = pygame.Rect(alien["pos"].x - self.ALIEN_WIDTH / 2, alien["pos"].y - self.ALIEN_HEIGHT / 2, self.ALIEN_WIDTH, self.ALIEN_HEIGHT)
                if proj_rect.colliderect(alien_rect):
                    # SFX: explosion
                    self.score += alien["points"] * self.combo_multiplier
                    reward += 0.1 + (1.0 * self.combo_multiplier)
                    
                    self._create_explosion(alien["pos"], alien["color"])
                    
                    self.aliens.remove(alien)
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    
                    self.combo_multiplier += 1
                    self.combo_timer = self.COMBO_TIMEOUT
                    break

        # Alien projectiles vs Player
        for proj in self.alien_projectiles[:]:
            proj_rect = pygame.Rect(proj.x - 3, proj.y - 3, 6, 6)
            if player_rect.colliderect(proj_rect):
                # SFX: player_hit
                self.alien_projectiles.remove(proj)
                if self.combo_multiplier > 1:
                    # SFX: combo_break
                    self.combo_multiplier = 1
                    self.combo_timer = 0
                reward -= 1.0 # Penalty for getting hit
                self._create_hit_effect(self.player_pos)

        return reward

    def _update_difficulty(self):
        self.alien_base_speed = 1.0 + (self.score // 500) * 0.1
        self.alien_fire_prob = 0.01 + (self.score // 1000) * 0.001

    def _create_explosion(self, pos, color):
        num_particles = min(20 + self.combo_multiplier * 2, 60)
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5 + self.combo_multiplier * 0.2)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": random.randint(20, 40),
                "radius": random.uniform(2, 6),
                "color": random.choice([color, (255, 255, 255), (255, 192, 0)])
            })

    def _create_hit_effect(self, pos):
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": random.randint(10, 20),
                "radius": random.uniform(1, 3),
                "color": self.COLOR_PLAYER
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for star in self.stars:
            c = star["brightness"]
            pygame.draw.rect(self.screen, (c, c, c), (star["pos"].x, star["pos"].y, star["size"], star["size"]))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 40.0))
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), max(0, int(p["radius"])), color)

        # Aliens
        for alien in self.aliens:
            pos = (int(alien["pos"].x), int(alien["pos"].y))
            color = alien["color"]
            pts = [
                (pos[0], pos[1] - self.ALIEN_HEIGHT / 2),
                (pos[0] - self.ALIEN_WIDTH / 2, pos[1] + self.ALIEN_HEIGHT / 2),
                (pos[0] + self.ALIEN_WIDTH / 2, pos[1] + self.ALIEN_HEIGHT / 2)
            ]
            pygame.gfxdraw.aapolygon(self.screen, pts, color)
            pygame.gfxdraw.filled_polygon(self.screen, pts, color)

        # Player Projectiles
        for proj in self.player_projectiles:
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE_PLAYER, (proj.x, proj.y), (proj.x, proj.y + 15), 3)

        # Alien Projectiles
        for proj in self.alien_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj.x), int(proj.y), 4, self.COLOR_PROJECTILE_ALIEN)
            pygame.gfxdraw.aacircle(self.screen, int(proj.x), int(proj.y), 4, self.COLOR_PROJECTILE_ALIEN)

        # Player
        px, py = int(self.player_pos.x), int(self.player_pos.y)
        player_pts = [
            (px, py - self.PLAYER_HEIGHT / 2),
            (px - self.PLAYER_WIDTH / 2, py + self.PLAYER_HEIGHT / 2),
            (px + self.PLAYER_WIDTH / 2, py + self.PLAYER_HEIGHT / 2)
        ]
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, px, py, int(self.PLAYER_WIDTH * 0.8), self.COLOR_PLAYER_GLOW)
        # Ship
        pygame.gfxdraw.aapolygon(self.screen, player_pts, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, player_pts, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Combo
        if self.combo_multiplier > 1:
            combo_text = f"x{self.combo_multiplier}"
            combo_surf = self.font_combo.render(combo_text, True, self.COLOR_TEXT)
            
            # Animate combo text size
            time_ratio = self.combo_timer / self.COMBO_TIMEOUT
            scale = 1.0 + 0.5 * (1 - time_ratio) # Pulses when it resets
            if self.combo_timer > self.COMBO_TIMEOUT - 10: # Pop in when new
                scale = 1.0 + 0.5 * (10 - (self.COMBO_TIMEOUT - self.combo_timer)) / 10.0
            
            scaled_surf = pygame.transform.smoothscale(combo_surf, (int(combo_surf.get_width() * scale), int(combo_surf.get_height() * scale)))
            
            text_rect = scaled_surf.get_rect(center=(self.SCREEN_WIDTH / 2, 35))
            self.screen.blit(scaled_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "combo": self.combo_multiplier,
            "aliens_remaining": len(self.aliens)
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Testing ---
    # This block will not run in a headless environment but is useful for local testing.
    # It requires a display to be available.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Combo Invaders")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # --- Game Loop ---
    while not terminated:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        if terminated and (info.get("aliens_remaining", 0) > 0 and info.get("steps", 0) < env.MAX_STEPS):
             print(f"Game Quit! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
        elif terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Reset for another round
            obs, info = env.reset()
            terminated = False
            total_reward = 0

        clock.tick(GameEnv.FPS)

    env.close()