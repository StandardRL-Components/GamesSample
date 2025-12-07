
# Generated: 2025-08-27T16:26:38.040104
# Source Brief: brief_01228.md
# Brief Index: 1228

        
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
        "Controls: ←→ to move. Hold space to fire your weapon."
    )

    game_description = (
        "A minimalist, procedurally generated top-down space shooter. "
        "Destroy waves of descending aliens to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # --- Game Constants ---
        self.MAX_STEPS = 10000
        self.TOTAL_WAVES = 5
        self.PLAYER_SPEED = 8
        self.PLAYER_FIRE_COOLDOWN = 6  # frames
        self.PLAYER_MAX_HEALTH = 3
        self.PROJECTILE_SPEED = 12
        self.ALIEN_BASE_V_SPEED = 0.5

        # --- Colors ---
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_ALIEN = (255, 64, 128)
        self.COLOR_PLAYER_PROJ = (200, 255, 255)
        self.COLOR_ALIEN_PROJ = (255, 200, 0)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_HEART = (255, 80, 80)
        
        # --- Game State ---
        # These are initialized here to satisfy linters, but are properly set in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.player_pos = [0, 0]
        self.player_health = 0
        self.player_fire_timer = 0
        self.player_hit_timer = 0
        self.current_wave = 0
        self.aliens = []
        self.projectiles = []
        self.particles = []
        self.starfield = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        self.player_pos = [self.screen_width // 2, self.screen_height - 50]
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_fire_timer = 0
        self.player_hit_timer = 0
        
        self.aliens = []
        self.projectiles = []
        self.particles = []

        self.current_wave = 1
        self._spawn_wave()
        
        # Create a static starfield for parallax effect
        self.starfield = []
        for _ in range(150):
            self.starfield.append({
                "pos": [self.np_random.uniform(0, self.screen_width), self.np_random.uniform(0, self.screen_height)],
                "size": self.np_random.uniform(1, 2.5),
                "speed_mult": self.np_random.uniform(0.1, 0.4)
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing and return the final state
            reward = 0
            terminated = True
            return (
                self._get_observation(),
                reward,
                terminated,
                False,
                self._get_info()
            )

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        step_reward = 0.1 # Survival reward

        # --- Update Game Logic ---
        self._handle_player_input(movement, space_held)
        self._update_aliens()
        self._update_projectiles()
        self._update_particles()
        
        collision_reward = self._handle_collisions()
        step_reward += collision_reward

        # --- Check for Inactivity ---
        if movement == 0 and not space_held:
            step_reward -= 0.01 # Small penalty for doing nothing

        # --- Wave Progression ---
        if not self.aliens and not self.victory:
            self.current_wave += 1
            if self.current_wave > self.TOTAL_WAVES:
                self.victory = True
                self.game_over = True
            else:
                self._spawn_wave()

        # --- Termination Conditions ---
        terminated = False
        if self.player_health <= 0:
            step_reward -= 100 # Loss penalty
            self.game_over = True
            terminated = True
        elif self.victory:
            step_reward += 100 # Victory bonus
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held):
        # Movement
        if movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.screen_width - 20)

        # Shooting
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1
        
        if space_held and self.player_fire_timer == 0:
            # sfx: player_shoot.wav
            self.projectiles.append({
                "pos": [self.player_pos[0], self.player_pos[1] - 20],
                "vel": [0, -self.PROJECTILE_SPEED],
                "type": "player"
            })
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_aliens(self):
        alien_v_speed = self.ALIEN_BASE_V_SPEED + (self.current_wave * 0.1)
        alien_fire_prob = (0.001 + self.current_wave * 0.0005)

        for alien in self.aliens:
            alien["pos"][1] += alien_v_speed
            alien["pos"][0] = alien["base_x"] + math.sin((self.steps + alien["offset"]) * 0.05) * 50

            if self.np_random.random() < alien_fire_prob:
                # sfx: alien_shoot.wav
                self.projectiles.append({
                    "pos": list(alien["pos"]),
                    "vel": [0, self.PROJECTILE_SPEED * 0.5],
                    "type": "alien"
                })

            if alien["pos"][1] > self.screen_height + 20:
                self.aliens.remove(alien)

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj["pos"][0] += proj["vel"][0]
            proj["pos"][1] += proj["vel"][1]
            if not (0 < proj["pos"][1] < self.screen_height):
                self.projectiles.remove(proj)
    
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 10, 20, 20)

        for proj in self.projectiles[:]:
            # Alien projectiles hitting player
            if proj["type"] == "alien" and self.player_hit_timer == 0:
                if player_rect.collidepoint(proj["pos"]):
                    # sfx: player_hit.wav
                    self.player_health -= 1
                    self.player_hit_timer = 30 # Invincibility frames
                    reward -= 1
                    self.projectiles.remove(proj)
                    self._create_explosion(self.player_pos, 20, self.COLOR_PLAYER)
                    if self.player_health <= 0:
                        # sfx: player_explosion.wav
                        self._create_explosion(self.player_pos, 100, self.COLOR_PLAYER)
                    continue
            
            # Player projectiles hitting aliens
            if proj["type"] == "player":
                for alien in self.aliens[:]:
                    alien_rect = pygame.Rect(alien["pos"][0] - 10, alien["pos"][1] - 10, 20, 20)
                    if alien_rect.collidepoint(proj["pos"]):
                        # sfx: alien_explosion.wav
                        self.aliens.remove(alien)
                        self.projectiles.remove(proj)
                        self.score += 10
                        reward += 10
                        self._create_explosion(alien["pos"], 30, self.COLOR_ALIEN)
                        break
        
        if self.player_hit_timer > 0:
            self.player_hit_timer -= 1
            
        return reward

    def _spawn_wave(self):
        num_aliens = 8 + self.current_wave * 2
        rows = math.ceil(num_aliens / 8)
        for i in range(num_aliens):
            row = i // 8
            col = i % 8
            self.aliens.append({
                "pos": [100 + col * 60, 50 + row * 50],
                "base_x": 100 + col * 60,
                "offset": i * 10
            })

    def _create_explosion(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(10, 25),
                "color": color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_starfield()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_starfield(self):
        for star in self.starfield:
            star["pos"][1] += star["speed_mult"]
            if star["pos"][1] > self.screen_height:
                star["pos"][1] = 0
                star["pos"][0] = self.np_random.uniform(0, self.screen_width)
            
            brightness = int(100 + 155 * star["speed_mult"])
            color = (brightness, brightness, brightness)
            pygame.draw.circle(self.screen, color, star["pos"], star["size"] / 2)

    def _render_game(self):
        # Render projectiles
        for proj in self.projectiles:
            color = self.COLOR_PLAYER_PROJ if proj["type"] == "player" else self.COLOR_ALIEN_PROJ
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, color)

        # Render aliens
        for alien in self.aliens:
            pos = (int(alien["pos"][0]), int(alien["pos"][1]))
            size = 10
            rect = (pos[0] - size, pos[1] - size, size * 2, size * 2)
            pygame.draw.rect(self.screen, self.COLOR_ALIEN, rect)

        # Render player
        if self.player_health > 0:
            is_invincible = self.player_hit_timer > 0
            if not (is_invincible and self.steps % 4 < 2): # Blink effect when hit
                p = self.player_pos
                points = [(p[0], p[1] - 15), (p[0] - 12, p[1] + 10), (p[0] + 12, p[1] + 10)]
                
                # Glow effect
                glow_points = [(p[0], p[1] - 20), (p[0] - 18, p[1] + 15), (p[0] + 18, p[1] + 15)]
                pygame.gfxdraw.aapolygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
                pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)

                # Main ship
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 25))
            color = p["color"] + (alpha,)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, 2, 2, 2, color)
            self.screen.blit(temp_surf, (int(p["pos"][0] - 2), int(p["pos"][1] - 2)))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_ui.render(f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}", True, self.COLOR_WHITE)
        self.screen.blit(wave_text, (self.screen_width - wave_text.get_width() - 10, 10))

        # Health
        for i in range(self.player_health):
            heart_pos = (20 + i * 30, 45)
            p1 = (heart_pos[0], heart_pos[1] + 5)
            p2 = (heart_pos[0] - 10, heart_pos[1] - 5)
            p3 = (heart_pos[0] + 10, heart_pos[1] - 5)
            p4 = (heart_pos[0], heart_pos[1] + 15)
            pygame.gfxdraw.bezier(self.screen, [p1, (p1[0]-10,p1[1]-10), (p2[0],p2[1]+5), p2], 10, self.COLOR_HEART)
            pygame.gfxdraw.bezier(self.screen, [p1, (p1[0]+10,p1[1]-10), (p3[0],p3[1]+5), p3], 10, self.COLOR_HEART)
        
        # Game Over / Victory Message
        if self.game_over:
            msg = "VICTORY" if self.victory else "GAME OVER"
            color = self.COLOR_PLAYER if self.victory else self.COLOR_ALIEN
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "health": self.player_health,
        }

    def validate_implementation(self):
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Space Shooter")
    screen_human = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    terminated = False
    running = True
    total_reward = 0
    
    while running:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        move_action = 0 # no-op
        if keys[pygame.K_LEFT]:
            move_action = 3
        elif keys[pygame.K_RIGHT]:
            move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move_action, space_action, shift_action]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated:
                print(f"Episode finished! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

        # --- Rendering for Human ---
        # The observation is already a rendered frame, so we just need to display it.
        # Pygame uses (width, height) but numpy uses (height, width), so we need to transpose.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_human.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for human play

    env.close()