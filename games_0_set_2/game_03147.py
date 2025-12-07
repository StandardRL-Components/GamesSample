
# Generated: 2025-08-28T07:07:17.488806
# Source Brief: brief_03147.md
# Brief Index: 3147

        
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

    # Short, user-facing control string
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to fire. Dodge enemy bullets and destroy the alien horde."
    )

    # Short, user-facing description of the game
    game_description = (
        "Defend Earth from a descending alien horde in this fast-paced, top-down arcade shooter. "
        "Survive as long as you can and aim for a high score!"
    )

    # Frames auto-advance at 30fps for smooth, real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)

        # --- Game Constants ---
        self.FPS = 30
        self.MAX_STEPS = 10000
        self.PLAYER_SPEED = 6
        self.PLAYER_PROJECTILE_SPEED = 12
        self.PLAYER_FIRE_COOLDOWN_FRAMES = 8  # ~4 shots/sec
        self.ALIEN_PROJECTILE_SPEED = 5
        self.INITIAL_ALIEN_SHOTS_PER_SEC = 0.5

        # --- Colors ---
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (100, 255, 180)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_ALIEN_GLOW = (255, 120, 120)
        self.COLOR_PLAYER_PROJ = (150, 255, 255)
        self.COLOR_ALIEN_PROJ = (255, 255, 0)
        self.COLOR_EXPLOSION = [(255, 200, 0), (255, 100, 0), (200, 50, 0)]
        self.COLOR_TEXT = (220, 220, 220)
        self.STAR_COLORS = [(50, 50, 70), (100, 100, 120), (150, 150, 180)]

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.np_random = None
        self.player_pos = None
        self.player_health = None
        self.player_fire_cooldown = 0
        self.player_hit_timer = 0
        self.aliens = []
        self.initial_alien_count = 0
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        self.stars = []
        
        # Initialize state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        # Player state
        self.player_pos = pygame.Vector2(self.width / 2, self.height - 50)
        self.player_health = 3
        self.player_fire_cooldown = 0
        self.player_hit_timer = 0

        # Entity lists
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []

        # Create alien formation
        rows, cols = 4, 9
        for row in range(rows):
            for col in range(cols):
                x = 80 + col * 60
                y = 50 + row * 40
                self.aliens.append({
                    "pos": pygame.Vector2(x, y),
                    "size": 12,
                })
        self.initial_alien_count = len(self.aliens)

        # Create starfield background
        self.stars = []
        for i in range(200):
            layer = self.np_random.integers(0, 3)
            self.stars.append({
                "pos": pygame.Vector2(self.np_random.uniform(0, self.width), self.np_random.uniform(0, self.height)),
                "size": 1 + layer,
                "speed": 0.5 + layer * 0.5
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If game is over, only advance frame and return terminal state
            self.clock.tick(self.FPS)
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()
            
        reward = -0.01  # Small penalty for each step to encourage efficiency

        self._handle_input(action)
        self._update_player()
        self._update_aliens()
        self._update_projectiles()
        reward += self._handle_collisions()
        self._update_particles()
        self._update_background()
        
        self.steps += 1
        
        # Check termination conditions
        terminated = False
        if self.player_health <= 0:
            terminated = True
            self.game_over = True
            self.victory = False
            reward += -100
            # sfx: player_explosion
        elif not self.aliens:
            terminated = True
            self.game_over = True
            self.victory = True
            reward += 100
            # sfx: victory_fanfare
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.victory = False

        if terminated and not self.victory:
            self._create_explosion(self.player_pos, 30, 40)

        self.clock.tick(self.FPS)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement == 1: self.player_pos.y -= self.PLAYER_SPEED # Up
        if movement == 2: self.player_pos.y += self.PLAYER_SPEED # Down
        if movement == 3: self.player_pos.x -= self.PLAYER_SPEED # Left
        if movement == 4: self.player_pos.x += self.PLAYER_SPEED # Right
        
        # Firing
        if space_held and self.player_fire_cooldown <= 0:
            proj_pos = self.player_pos + pygame.Vector2(0, -20)
            self.player_projectiles.append(proj_pos)
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN_FRAMES
            # sfx: player_shoot

    def _update_player(self):
        # Clamp player position to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, 20, self.width - 20)
        self.player_pos.y = np.clip(self.player_pos.y, self.height / 2, self.height - 20)

        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
        if self.player_hit_timer > 0:
            self.player_hit_timer -= 1

    def _update_aliens(self):
        # Alien descent and sway
        descent_speed = 0.1 + 0.5 * (1 - len(self.aliens) / self.initial_alien_count)
        sway = math.sin(self.steps / 60) * 0.5
        for alien in self.aliens:
            alien["pos"].y += descent_speed
            alien["pos"].x += sway
            if alien["pos"].y > self.height: # Alien escaped
                self.aliens.remove(alien)
                self.player_health -= 1 # Penalty for letting one pass
                self.player_hit_timer = 15 # Flash effect

        # Alien firing logic
        aliens_killed = self.initial_alien_count - len(self.aliens)
        shots_per_sec = self.INITIAL_ALIEN_SHOTS_PER_SEC + (aliens_killed // 10) * 0.1
        fire_prob = shots_per_sec / self.FPS

        if self.aliens and self.np_random.random() < fire_prob:
            shooter = self.np_random.choice(self.aliens)
            proj_pos = shooter["pos"].copy()
            self.alien_projectiles.append({"pos": proj_pos, "dodged": False})
            # sfx: alien_shoot

    def _update_projectiles(self):
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj.y -= self.PLAYER_PROJECTILE_SPEED
            if proj.y < 0:
                self.player_projectiles.remove(proj)
        
        # Alien projectiles
        for proj in self.alien_projectiles[:]:
            proj["pos"].y += self.ALIEN_PROJECTILE_SPEED
            if proj["pos"].y > self.height:
                self.alien_projectiles.remove(proj)

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        for proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                if proj.distance_to(alien["pos"]) < alien["size"]:
                    self.player_projectiles.remove(proj)
                    self.aliens.remove(alien)
                    self._create_explosion(alien["pos"], 15, 20)
                    reward += 1.0
                    self.score += 100
                    # sfx: alien_explosion
                    break
        
        # Alien projectiles vs Player
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 10, 20, 20)
        for proj in self.alien_projectiles[:]:
            if player_rect.collidepoint(proj["pos"]):
                self.alien_projectiles.remove(proj)
                if self.player_hit_timer <= 0:
                    self.player_health -= 1
                    self.player_hit_timer = 30 # Invincibility frames
                    self._create_explosion(self.player_pos, 10, 15)
                    # sfx: player_hit
                break
            # Dodge reward
            elif not proj["dodged"] and proj["pos"].y > self.player_pos.y:
                reward += 0.1
                proj["dodged"] = True
        
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                p["radius"] += p["expansion"]

    def _update_background(self):
        for star in self.stars:
            star["pos"].y += star["speed"]
            if star["pos"].y > self.height:
                star["pos"].y = 0
                star["pos"].x = self.np_random.uniform(0, self.width)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_projectiles()
        self._render_aliens()
        if self.player_health > 0:
            self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "aliens_remaining": len(self.aliens),
        }

    def _create_explosion(self, pos, num_particles, max_radius):
        for _ in range(num_particles):
            self.particles.append({
                "pos": pos.copy(),
                "life": self.np_random.integers(10, 20),
                "color": self.np_random.choice(self.COLOR_EXPLOSION),
                "radius": self.np_random.uniform(1, 3),
                "expansion": self.np_random.uniform(0.5, max_radius / 15)
            })
    
    # --- RENDER METHODS ---

    def _render_background(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, self.STAR_COLORS[star["size"]-1], star["pos"], star["size"])

    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Flash effect on hit
        if self.player_hit_timer > 0 and self.steps % 4 < 2:
            return

        # Player ship shape (chevron)
        points = [
            (pos[0], pos[1] - 15),
            (pos[0] + 12, pos[1] + 10),
            (pos[0], pos[1]),
            (pos[0] - 12, pos[1] + 10),
        ]
        
        # Glow effect
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_aliens(self):
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 0.8 + 0.2
        for alien in self.aliens:
            pos = (int(alien["pos"].x), int(alien["pos"].y))
            size = int(alien["size"] * (0.8 + pulse * 0.2))
            
            # Simple alien shape
            rect = pygame.Rect(pos[0] - size, pos[1] - size // 2, size * 2, size)
            
            # Glow effect
            pygame.draw.ellipse(self.screen, self.COLOR_ALIEN_GLOW, rect.inflate(4, 4), 2)
            pygame.draw.ellipse(self.screen, self.COLOR_ALIEN, rect)

    def _render_projectiles(self):
        for proj in self.player_projectiles:
            start = (int(proj.x), int(proj.y))
            end = (int(proj.x), int(proj.y + 8))
            pygame.draw.line(self.screen, self.COLOR_PLAYER_PROJ, start, end, 3)

        for proj in self.alien_projectiles:
            pos = (int(proj["pos"].x), int(proj["pos"].y))
            pygame.draw.circle(self.screen, self.COLOR_ALIEN_PROJ, pos, 4)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, self.COLOR_ALIEN_PROJ)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20))
            color = (*p["color"], alpha)
            surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(surf, (int(p["pos"].x - p["radius"]), int(p["pos"].y - p["radius"])), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Health
        health_text = self.font_ui.render(f"HEALTH: {self.player_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.width - health_text.get_width() - 10, 10))
        
        # Game Over / Victory message
        if self.game_over:
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_PLAYER if self.victory else self.COLOR_ALIEN
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
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

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Alien Horde Defender")
    screen_display = pygame.display.set_mode((env.width, env.height))

    total_reward = 0
    total_steps = 0

    while not done:
        # --- Human Controls ---
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
        total_steps += 1

        # --- Render to Display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Game Over!")
    print(f"Final Score: {info['score']}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Steps: {total_steps}")
    
    env.close()