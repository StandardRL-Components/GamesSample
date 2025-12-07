
# Generated: 2025-08-27T23:11:24.215763
# Source Brief: brief_03378.md
# Brief Index: 3378

        
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
        "Controls: ←→ to move the ship. Press space to fire your weapon. Destroy all aliens to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down shooter where the player must destroy waves of descending aliens while dodging their projectiles."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.TOTAL_WAVES = 5

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_OUTLINE = (200, 255, 220)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_ALIEN_OUTLINE = (255, 150, 150)
        self.COLOR_PROJECTILE_PLAYER = (200, 255, 255)
        self.COLOR_PROJECTILE_ALIEN = (255, 200, 200)
        self.COLOR_EXPLOSION = [(255, 255, 0), (255, 128, 0), (255, 64, 0)]
        self.COLOR_UI = (200, 200, 255)
        self.COLOR_STAR = [(100, 100, 120), (150, 150, 180), (200, 200, 255)]

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Initialize state variables
        self.player_pos = None
        self.aliens = None
        self.player_projectiles = None
        self.alien_projectiles = None
        self.particles = None
        self.stars = None
        self.player_fire_cooldown = None
        self.steps = None
        self.score = None
        self.current_wave = None
        self.terminated = None
        self.win = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.current_wave = 1
        self.terminated = False
        self.win = False

        self.player_pos = [self.WIDTH / 2, self.HEIGHT - 40]
        self.player_fire_cooldown = 0

        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []

        self._setup_wave()
        
        if self.stars is None:
            self.stars = [
                (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.uniform(0.2, 1.0))
                for _ in range(100)
            ]

        return self._get_observation(), self._get_info()

    def _setup_wave(self):
        self.aliens.clear()
        num_aliens = 10 + self.current_wave * 2
        rows = 2 + self.current_wave // 2
        cols = math.ceil(num_aliens / rows)
        x_spacing = self.WIDTH * 0.8 / max(1, cols - 1) if cols > 1 else 0
        y_spacing = 40

        for r in range(rows):
            for c in range(cols):
                if len(self.aliens) < num_aliens:
                    x = self.WIDTH * 0.1 + c * x_spacing
                    y = 50 + r * y_spacing
                    fire_rate_base = 2.0 - (self.current_wave - 1) * 0.25
                    fire_rate = max(0.5, fire_rate_base)
                    self.aliens.append({
                        "pos": [x, y],
                        "fire_cooldown": self.FPS * random.uniform(0.5, fire_rate),
                        "radius": 12,
                    })

    def step(self, action):
        reward = -0.1  # Time penalty to encourage speed
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        if not self.terminated:
            self._handle_input(movement, space_held)
            self._update_game_objects()
            
            collision_reward = self._handle_collisions()
            reward += collision_reward

            bonus_reward = self._calculate_bonus_rewards(movement, space_held)
            reward += bonus_reward

            wave_reward = self._check_game_state()
            reward += wave_reward

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.terminated = True

        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        player_speed = 8
        if movement == 3:  # Left
            self.player_pos[0] -= player_speed
        if movement == 4:  # Right
            self.player_pos[0] += player_speed
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)

        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1

        if space_held and self.player_fire_cooldown <= 0:
            # Sound effect placeholder: # pew pew
            self.player_projectiles.append({"pos": list(self.player_pos), "vel": [0, -15]})
            self.player_fire_cooldown = 8  # Cooldown in frames (30fps)

    def _update_game_objects(self):
        # Update player projectiles
        for p in self.player_projectiles[:]:
            p["pos"][1] += p["vel"][1]
            if p["pos"][1] < 0:
                self.player_projectiles.remove(p)

        # Update alien projectiles
        for p in self.alien_projectiles[:]:
            p["pos"][1] += p["vel"][1]
            if p["pos"][1] > self.HEIGHT:
                self.alien_projectiles.remove(p)

        # Update aliens and their firing
        for alien in self.aliens:
            alien["fire_cooldown"] -= 1
            if alien["fire_cooldown"] <= 0:
                fire_rate_base = 2.0 - (self.current_wave - 1) * 0.25
                fire_rate = max(0.5, fire_rate_base)
                alien["fire_cooldown"] = self.FPS * random.uniform(0.5, fire_rate)
                self.alien_projectiles.append({"pos": list(alien["pos"]), "vel": [0, 5]})
        
        # Update particles
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
    
    def _handle_collisions(self):
        reward = 0
        # Player projectiles vs aliens
        for pp in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                dist = math.hypot(pp["pos"][0] - alien["pos"][0], pp["pos"][1] - alien["pos"][1])
                if dist < alien["radius"] + 4:
                    # Sound effect placeholder: # alien explosion
                    self._create_explosion(alien["pos"], self.COLOR_EXPLOSION, 20)
                    self.aliens.remove(alien)
                    if pp in self.player_projectiles: self.player_projectiles.remove(pp)
                    self.score += 10
                    reward += 10
                    break

        # Alien projectiles vs player
        player_radius = 15
        for ap in self.alien_projectiles[:]:
            dist = math.hypot(ap["pos"][0] - self.player_pos[0], ap["pos"][1] - self.player_pos[1])
            if dist < player_radius + 4:
                # Sound effect placeholder: # player explosion
                self._create_explosion(self.player_pos, self.COLOR_PLAYER, 40)
                self.terminated = True
                self.win = False
                reward -= 10
                break
        return reward

    def _calculate_bonus_rewards(self, movement, space_held):
        reward = 0
        if not self.aliens: return 0

        # +0.2 for moving towards alien cluster
        alien_center_x = sum(a["pos"][0] for a in self.aliens) / len(self.aliens)
        if movement == 3 and self.player_pos[0] > alien_center_x:
            reward += 0.2
        elif movement == 4 and self.player_pos[0] < alien_center_x:
            reward += 0.2
            
        # +2 for risky shot (firing near enemy projectile)
        if space_held:
            for ap in self.alien_projectiles:
                if math.hypot(ap["pos"][0] - self.player_pos[0], ap["pos"][1] - self.player_pos[1]) < 80:
                    reward += 2
                    break
        
        # -2 for safe play (moving away when aliens are firing)
        is_firing_nearby = any(math.hypot(a["pos"][0] - self.player_pos[0], a["pos"][1] - self.player_pos[1]) < 100 and a['fire_cooldown'] < 5 for a in self.aliens)
        if is_firing_nearby:
            if movement == 3 and self.player_pos[0] < alien_center_x:
                reward -= 2
            elif movement == 4 and self.player_pos[0] > alien_center_x:
                reward -= 2
        
        return reward

    def _check_game_state(self):
        reward = 0
        if not self.aliens and not self.terminated:
            if self.current_wave < self.TOTAL_WAVES:
                self.current_wave += 1
                self._setup_wave()
                self.score += 100
                reward += 100
            else:
                self.terminated = True
                self.win = True
                self.score += 100
                reward += 100
        return reward

    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": random.randint(10, 20),
                "color": random.choice(color) if isinstance(color, list) else color,
                "radius": random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_aliens()
        self._render_projectiles()
        self._render_particles()
        if not (self.terminated and not self.win):
            self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "win": self.win
        }

    def _render_stars(self):
        for i in range(len(self.stars)):
            x, y, speed = self.stars[i]
            y = (y + speed) % self.HEIGHT
            self.stars[i] = (x, y, speed)
            color_val = int(speed * 150) + 50
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (x, y), speed)

    def _render_player(self):
        x, y = self.player_pos
        points = [
            (x, y - 15),
            (x - 12, y + 10),
            (x + 12, y + 10)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        
        # Engine glow
        if self.player_fire_cooldown > 4: # Glow when firing
            glow_y = y + 15
            pygame.gfxdraw.aacircle(self.screen, int(x), int(glow_y), 5, self.COLOR_PLAYER_OUTLINE)
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(glow_y), 4, self.COLOR_PLAYER)


    def _render_aliens(self):
        for alien in self.aliens:
            x, y = alien["pos"]
            r = alien["radius"]
            # Pulsing effect
            pulse = abs(math.sin(self.steps * 0.1 + x)) * 3
            pygame.gfxdraw.aacircle(self.screen, int(x), int(y), int(r + pulse), self.COLOR_ALIEN_OUTLINE)
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), int(r - 1 + pulse), self.COLOR_ALIEN)

    def _render_projectiles(self):
        for p in self.player_projectiles:
            x, y = p["pos"]
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE_PLAYER, (x - 2, y - 8, 4, 16))
        for p in self.alien_projectiles:
            x, y = p["pos"]
            pygame.gfxdraw.aacircle(self.screen, int(x), int(y), 4, self.COLOR_PROJECTILE_ALIEN)
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), 3, self.COLOR_PROJECTILE_ALIEN)
            
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20))
            color = p["color"] + (alpha,)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (p["pos"][0] - p["radius"], p["pos"][1] - p["radius"]))

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        wave_text = self.font_ui.render(f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}", True, self.COLOR_UI)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        if self.terminated:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.COLOR_ALIEN
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
            
    def close(self):
        pygame.quit()

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
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a persistent pygame screen for display
    pygame.display.set_caption("Galactic Annihilator")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # Action defaults to no-op
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Map keys to actions
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1

        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        if terminated:
            # If the game is over, wait for a key press to reset
            if any(keys):
                obs, info = env.reset()
                terminated = False
                total_reward = 0
        else:
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation to the display screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(env.FPS)

    env.close()