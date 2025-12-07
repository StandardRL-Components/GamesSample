
# Generated: 2025-08-28T06:07:21.640435
# Source Brief: brief_02843.md
# Brief Index: 2843

        
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
    """
    A top-down arcade shooter where the player must destroy waves of procedurally 
    generated alien ships while dodging their fire.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys to move. Press space to fire."
    )

    # User-facing description of the game
    game_description = (
        "A top-down shooter where you destroy waves of alien ships while dodging their fire."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and rendering setup
        self.width, self.height = 640, 400
        self.render_mode = render_mode
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.MAX_STEPS = 5000
        self.TOTAL_ALIENS_TO_WIN = 50
        self.PLAYER_SPEED = 6
        self.PLAYER_PROJECTILE_SPEED = 10
        self.ENEMY_PROJECTILE_SPEED = 4
        self.PLAYER_FIRE_COOLDOWN_MAX = 8  # Frames between shots
        self.PLAYER_INVINCIBILITY_DURATION = 90 # Frames after being hit

        # Colors
        self.COLOR_BG = (10, 0, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_PLAYER_PROJECTILE = (200, 255, 255)
        self.COLOR_ENEMY_PROJECTILE = (255, 200, 50)
        self.COLOR_EXPLOSION = (255, 255, 0)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_STAR = (100, 100, 120)

        # Fonts
        try:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_game_over = pygame.font.Font(None, 72)
        except IOError:
            self.font_ui = pygame.font.SysFont("sans", 24)
            self.font_game_over = pygame.font.SysFont("sans", 72)

        # Initialize state variables
        self.player_pos = None
        self.player_lives = None
        self.player_projectiles = None
        self.enemy_projectiles = None
        self.aliens = None
        self.explosions = None
        self.stars = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.current_stage = None
        self.aliens_destroyed_total = None
        self.player_hit_cooldown = None
        self.player_fire_cooldown = None
        
        self.validate_implementation()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.player_pos = pygame.Vector2(self.width / 2, self.height - 50)
        self.player_lives = 3
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.aliens = []
        self.explosions = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.current_stage = 0
        self.aliens_destroyed_total = 0
        
        self.player_hit_cooldown = 0
        self.player_fire_cooldown = 0
        
        # Create a static starfield for the background
        self.stars = [
            (self.np_random.integers(0, self.width), self.np_random.integers(0, self.height))
            for _ in range(100)
        ]
        
        self._spawn_stage()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, no actions have an effect.
            # Return the final state.
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Small penalty for each step to encourage speed

        # --- UPDATE GAME STATE ---
        self._handle_input(action)
        self._update_player()
        self._update_projectiles()
        self._update_aliens()
        self._update_explosions()
        reward += self._handle_collisions()
        
        self._check_stage_clear()
        
        self.steps += 1
        
        # --- CHECK TERMINATION ---
        terminated = False
        if self.player_lives <= 0:
            self.game_over = True
            terminated = True
        elif self.aliens_destroyed_total >= self.TOTAL_ALIENS_TO_WIN:
            self.game_over = True
            self.game_won = True
            terminated = True
            reward += 100 # Bonus for winning
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Player movement
        if movement == 1: # Up
            self.player_pos.y -= self.PLAYER_SPEED
        if movement == 2: # Down
            self.player_pos.y += self.PLAYER_SPEED
        if movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
        if movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
            
        # Clamp player position to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, 15, self.width - 15)
        self.player_pos.y = np.clip(self.player_pos.y, 15, self.height - 15)
        
        # Player firing
        if space_held and self.player_fire_cooldown == 0:
            # Sound: Player Shoot
            projectile_pos = self.player_pos + pygame.Vector2(0, -20)
            self.player_projectiles.append(pygame.Rect(projectile_pos.x - 2, projectile_pos.y, 4, 10))
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN_MAX

    def _update_player(self):
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
        if self.player_hit_cooldown > 0:
            self.player_hit_cooldown -= 1
            
    def _update_projectiles(self):
        # Move player projectiles
        for proj in self.player_projectiles[:]:
            proj.y -= self.PLAYER_PROJECTILE_SPEED
            if proj.bottom < 0:
                self.player_projectiles.remove(proj)
        
        # Move enemy projectiles
        for proj in self.enemy_projectiles[:]:
            proj.y += self.ENEMY_PROJECTILE_SPEED
            if proj.top > self.height:
                self.enemy_projectiles.remove(proj)

    def _update_aliens(self):
        for alien in self.aliens:
            # Move alien based on its pattern
            time = self.steps + alien["pattern_params"]["offset"]
            if alien["pattern"] == "sinusoidal":
                alien["rect"].centerx = alien["pattern_params"]["center_x"] + \
                    math.sin(time * alien["pattern_params"]["freq"]) * alien["pattern_params"]["amp"]
            elif alien["pattern"] == "circular":
                alien["rect"].centerx = alien["pattern_params"]["center_x"] + \
                    math.cos(time * alien["pattern_params"]["freq"]) * alien["pattern_params"]["amp"]
                alien["rect"].centery = alien["pattern_params"]["center_y"] + \
                    math.sin(time * alien["pattern_params"]["freq"]) * alien["pattern_params"]["amp"]

            # Alien firing
            alien["fire_cooldown"] -= 1
            if alien["fire_cooldown"] <= 0:
                # Sound: Enemy Shoot
                proj_pos = pygame.Vector2(alien["rect"].centerx, alien["rect"].bottom)
                self.enemy_projectiles.append(pygame.Rect(proj_pos.x - 2, proj_pos.y, 4, 10))
                alien["fire_cooldown"] = alien["fire_rate"] + self.np_random.integers(-10, 10)

    def _update_explosions(self):
        for explosion in self.explosions[:]:
            explosion["radius"] += explosion["speed"]
            if explosion["radius"] > explosion["max_radius"]:
                self.explosions.remove(explosion)

    def _handle_collisions(self):
        collision_reward = 0
        
        # Player projectiles vs aliens
        for proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                if alien["rect"].colliderect(proj):
                    # Sound: Explosion
                    self.explosions.append({
                        "pos": pygame.Vector2(alien["rect"].center),
                        "radius": 5, "max_radius": 30, "speed": 2
                    })
                    self.aliens.remove(alien)
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    
                    self.score += 10
                    self.aliens_destroyed_total += 1
                    collision_reward += 10
                    break
        
        # Enemy projectiles vs player
        if self.player_hit_cooldown == 0:
            player_rect = pygame.Rect(self.player_pos.x - 12, self.player_pos.y - 10, 24, 20)
            for proj in self.enemy_projectiles[:]:
                if player_rect.colliderect(proj):
                    # Sound: Player Hit
                    self.enemy_projectiles.remove(proj)
                    self.player_lives -= 1
                    self.player_hit_cooldown = self.PLAYER_INVINCIBILITY_DURATION
                    collision_reward -= 10
                    self.explosions.append({
                        "pos": self.player_pos.copy(),
                        "radius": 10, "max_radius": 50, "speed": 2
                    })
                    break
        return collision_reward

    def _spawn_stage(self):
        self.current_stage += 1
        
        aliens_per_row = 8
        rows = 0
        
        if self.current_stage == 1:
            rows = 2 # 16 aliens
        elif self.current_stage == 2:
            rows = 2 # 16 aliens
        elif self.current_stage == 3:
            rows = 3 # 18 aliens
        
        base_fire_rate = max(20, 35 - self.current_stage * 5)
        
        for r in range(rows):
            for c in range(aliens_per_row):
                if self.aliens_destroyed_total + len(self.aliens) >= self.TOTAL_ALIENS_TO_WIN:
                    break

                center_x = (self.width / (aliens_per_row + 1)) * (c + 1)
                center_y = 60 + r * 50
                
                alien = {
                    "rect": pygame.Rect(center_x - 12, center_y - 12, 24, 24),
                    "fire_cooldown": self.np_random.integers(30, 100),
                    "fire_rate": base_fire_rate,
                }
                
                if self.current_stage == 1: # Simple sinusoidal wave
                    alien["pattern"] = "sinusoidal"
                    alien["pattern_params"] = {
                        "center_x": center_x, "amp": 50, "freq": 0.02,
                        "offset": c * 10
                    }
                else: # Circular pattern for later stages
                    alien["pattern"] = "circular"
                    alien["pattern_params"] = {
                        "center_x": center_x, "center_y": center_y,
                        "amp": 30 + r * 5, "freq": 0.03, "offset": c * 15
                    }
                self.aliens.append(alien)

    def _check_stage_clear(self):
        if not self.aliens and not self.game_won:
            self._spawn_stage()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "stage": self.current_stage,
            "aliens_left": self.TOTAL_ALIENS_TO_WIN - self.aliens_destroyed_total,
        }

    def _render_game(self):
        # Draw stars
        for star_pos in self.stars:
            self.screen.set_at(star_pos, self.COLOR_STAR)
            
        # Draw player projectiles
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJECTILE, proj, border_radius=2)
            
        # Draw enemy projectiles
        for proj in self.enemy_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_PROJECTILE, proj, border_radius=2)
            
        # Draw aliens
        for alien in self.aliens:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, alien["rect"], border_radius=4)
            # Add a simple "engine" glow
            glow_rect = alien["rect"].copy()
            glow_rect.height = 4
            glow_rect.top = alien["rect"].bottom
            pygame.draw.rect(self.screen, (255, 150, 0), glow_rect, border_radius=2)

        # Draw player
        if self.player_lives > 0:
            # Flicker when invincible
            if self.player_hit_cooldown == 0 or self.steps % 4 < 2:
                p1 = self.player_pos + pygame.Vector2(0, -15)
                p2 = self.player_pos + pygame.Vector2(-12, 10)
                p3 = self.player_pos + pygame.Vector2(12, 10)
                
                # Draw a slightly larger, semi-transparent triangle for a glow effect
                pygame.gfxdraw.aatrigon(self.screen, int(p1.x), int(p1.y), int(p2.x), int(p2.y), int(p3.x), int(p3.y), self.COLOR_PLAYER)
                pygame.gfxdraw.filled_trigon(self.screen, int(p1.x), int(p1.y), int(p2.x), int(p2.y), int(p3.x), int(p3.y), self.COLOR_PLAYER)
        
        # Draw explosions
        for explosion in self.explosions:
            # Draw multiple circles for a better effect
            alpha = max(0, 255 - int((explosion["radius"] / explosion["max_radius"]) * 255))
            color = (*self.COLOR_EXPLOSION, alpha)
            
            # Use a surface for transparency
            s = pygame.Surface((explosion["radius"]*2, explosion["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (explosion["radius"], explosion["radius"]), explosion["radius"])
            self.screen.blit(s, (explosion["pos"].x - explosion["radius"], explosion["pos"].y - explosion["radius"]))

    def _render_ui(self):
        # Draw score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        # Draw lives
        for i in range(self.player_lives):
            p1 = (self.width - 30 - i * 25, 15)
            p2 = (self.width - 40 - i * 25, 30)
            p3 = (self.width - 20 - i * 25, 30)
            pygame.gfxdraw.filled_trigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_PLAYER)

        # Draw game over/win message
        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (0, 255, 0) if self.game_won else (255, 0, 0)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.width / 2, self.height / 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, bg_rect.topleft)
            
            self.screen.blit(end_text, text_rect)
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.height, self.width, 3)
        assert obs.dtype == np.uint8
        
        # Test info
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # To display the game, we need to create a window
    pygame.display.set_caption("Arcade Shooter")
    screen = pygame.display.set_mode((env.width, env.height))
    
    terminated = False
    total_reward = 0
    
    # --- Manual Control ---
    # To play manually, uncomment the following block and comment out the random agent block.
    # running = True
    # while running:
    #     movement = 0 # no-op
    #     space = 0
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]: movement = 1
    #     if keys[pygame.K_DOWN]: movement = 2
    #     if keys[pygame.K_LEFT]: movement = 3
    #     if keys[pygame.K_RIGHT]: movement = 4
    #     if keys[pygame.K_SPACE]: space = 1
        
    #     action = [movement, space, 0] # Shift is not used
        
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     total_reward += reward
        
    #     # Render to the screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #         if terminated and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
    #             obs, info = env.reset()
    #             terminated = False
    #             total_reward = 0

    #     env.clock.tick(30) # Run at 30 FPS
        
    # --- Random Agent ---
    for _ in range(1000):
        if terminated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        
        env.clock.tick(30) # Run at 30 FPS
        
    env.close()