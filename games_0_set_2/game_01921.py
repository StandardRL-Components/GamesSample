
# Generated: 2025-08-28T03:07:09.009839
# Source Brief: brief_01921.md
# Brief Index: 1921

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move. Hold space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist top-down space shooter. Destroy all invading aliens to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.TOTAL_ALIENS = 50

        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_ALIEN = (220, 50, 50)
        self.COLOR_PLAYER_PROJ = (255, 255, 255)
        self.COLOR_ALIEN_PROJ = (50, 220, 50)
        self.COLOR_EXPLOSION = (255, 255, 255)
        self.COLOR_UI = (200, 200, 200)

        # Player settings
        self.PLAYER_WIDTH, self.PLAYER_HEIGHT = 30, 15
        self.PLAYER_SPEED = 8
        self.PLAYER_FIRE_COOLDOWN_MAX = 8 # frames
        self.PLAYER_RESPAWN_INVULNERABILITY = 60 # frames

        # Alien settings
        self.ALIEN_COLS, self.ALIEN_ROWS = 10, 5
        self.ALIEN_SIZE = 20
        self.ALIEN_H_SPACING = 45
        self.ALIEN_V_SPACING = 30
        self.ALIEN_H_SPEED = 1
        self.ALIEN_V_DROP = 10
        self.INITIAL_ALIEN_FIRE_PROB = 0.01
        self.ALIEN_FIRE_PROB_INCREASE = 0.0008

        # Projectile settings
        self.PROJ_SPEED = 10
        self.PROJ_RADIUS = 3

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_lives = 0
        self.player_pos = None
        self.player_fire_cooldown = 0
        self.player_invulnerable_timer = 0
        self.aliens = []
        self.alien_direction = 1
        self.alien_fire_prob = 0.0
        self.player_projectiles = []
        self.alien_projectiles = []
        self.explosions = []

        # Initialize state for the first time
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_lives = 3
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        self.player_fire_cooldown = 0
        self.player_invulnerable_timer = 0

        self.aliens = []
        start_x = (self.WIDTH - (self.ALIEN_COLS - 1) * self.ALIEN_H_SPACING) / 2
        start_y = 50
        for row in range(self.ALIEN_ROWS):
            for col in range(self.ALIEN_COLS):
                alien = {
                    "rect": pygame.Rect(
                        start_x + col * self.ALIEN_H_SPACING,
                        start_y + row * self.ALIEN_V_SPACING,
                        self.ALIEN_SIZE,
                        self.ALIEN_SIZE
                    ),
                    "id": row * self.ALIEN_COLS + col
                }
                self.aliens.append(alien)
        
        self.alien_direction = 1
        self.alien_fire_prob = self.INITIAL_ALIEN_FIRE_PROB
        
        self.player_projectiles = []
        self.alien_projectiles = []
        self.explosions = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        movement, space_held, _ = action
        space_pressed = space_held == 1
        
        reward = -0.01 # Time penalty
        terminated = False
        
        if not self.game_over:
            # --- Update Game Logic ---
            self._handle_input(movement, space_pressed)
            self._update_player()
            self._update_projectiles()
            self._update_aliens()
            reward += self._handle_collisions()
            self._update_explosions()

            # --- Check Termination Conditions ---
            if self.player_lives <= 0:
                self.game_over = True
                terminated = True
                reward -= 100
                # sfx: game_over_sound
            elif not self.aliens:
                self.game_over = True
                terminated = True
                reward += 100
                # sfx: victory_sound
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # truncated is always False
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        # Player Movement
        if movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
        
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_WIDTH / 2, self.WIDTH - self.PLAYER_WIDTH / 2)

        # Player Firing
        if space_pressed and self.player_fire_cooldown <= 0 and self.player_lives > 0:
            proj_pos = pygame.Vector2(self.player_pos.x, self.player_pos.y - self.PLAYER_HEIGHT)
            self.player_projectiles.append(proj_pos)
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN_MAX
            # sfx: player_shoot

    def _update_player(self):
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
        if self.player_invulnerable_timer > 0:
            self.player_invulnerable_timer -= 1
            
    def _update_projectiles(self):
        # Move player projectiles and remove off-screen
        self.player_projectiles[:] = [p for p in self.player_projectiles if p.y > 0]
        for p in self.player_projectiles:
            p.y -= self.PROJ_SPEED
        
        # Move alien projectiles and remove off-screen
        self.alien_projectiles[:] = [p for p in self.alien_projectiles if p.y < self.HEIGHT]
        for p in self.alien_projectiles:
            p.y += self.PROJ_SPEED

    def _update_aliens(self):
        if not self.aliens:
            return

        # Horizontal movement and boundary check
        move_down = False
        for alien in self.aliens:
            alien["rect"].x += self.ALIEN_H_SPEED * self.alien_direction
            if alien["rect"].left < 0 or alien["rect"].right > self.WIDTH:
                move_down = True
        
        # Vertical movement
        if move_down:
            self.alien_direction *= -1
            for alien in self.aliens:
                alien["rect"].y += self.ALIEN_V_DROP
                # sfx: alien_move_down
        
        # Alien Firing
        # Find aliens eligible to fire (bottom of each column)
        eligible_firers = {}
        for alien in self.aliens:
            col = round((alien["rect"].centerx - self.aliens[0]["rect"].centerx) / self.ALIEN_H_SPACING) if self.aliens else 0
            if col not in eligible_firers or alien["rect"].bottom > eligible_firers[col]["rect"].bottom:
                eligible_firers[col] = alien
        
        if eligible_firers and self.np_random.random() < self.alien_fire_prob:
            firing_alien = self.np_random.choice(list(eligible_firers.values()))
            proj_pos = pygame.Vector2(firing_alien["rect"].centerx, firing_alien["rect"].bottom)
            self.alien_projectiles.append(proj_pos)
            # sfx: alien_shoot

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(0, 0, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        player_rect.center = self.player_pos

        # Player projectiles vs Aliens
        for proj in self.player_projectiles[:]:
            collided = False
            for alien in self.aliens[:]:
                if alien["rect"].collidepoint(proj):
                    self.player_projectiles.remove(proj)
                    self.aliens.remove(alien)
                    self.score += 10
                    reward += 1
                    self.alien_fire_prob += self.ALIEN_FIRE_PROB_INCREASE
                    self._create_explosion(alien["rect"].center)
                    # sfx: alien_explosion
                    collided = True
                    break
            if collided:
                break
        
        # Alien projectiles vs Player
        if self.player_invulnerable_timer <= 0 and self.player_lives > 0:
            for proj in self.alien_projectiles[:]:
                if player_rect.collidepoint(proj):
                    self.alien_projectiles.remove(proj)
                    self.player_lives -= 1
                    self._create_explosion(self.player_pos, radius=30, duration=30)
                    self.player_invulnerable_timer = self.PLAYER_RESPAWN_INVULNERABILITY
                    self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40) # Respawn
                    # sfx: player_explosion
                    break
        return reward

    def _create_explosion(self, pos, radius=15, duration=15):
        self.explosions.append({"pos": pos, "radius": radius, "timer": duration, "max_timer": duration})

    def _update_explosions(self):
        self.explosions[:] = [e for e in self.explosions if e["timer"] > 0]
        for e in self.explosions:
            e["timer"] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw aliens
        for alien in self.aliens:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN, alien["rect"])

        # Draw player
        if self.player_lives > 0:
            is_invulnerable = self.player_invulnerable_timer > 0
            if not is_invulnerable or (is_invulnerable and self.steps % 10 < 5):
                player_rect = pygame.Rect(0, 0, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
                player_rect.center = self.player_pos
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # Draw projectiles
        for p in self.player_projectiles:
            pygame.gfxdraw.aacircle(self.screen, int(p.x), int(p.y), self.PROJ_RADIUS, self.COLOR_PLAYER_PROJ)
            pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), self.PROJ_RADIUS, self.COLOR_PLAYER_PROJ)
        for p in self.alien_projectiles:
            pygame.gfxdraw.aacircle(self.screen, int(p.x), int(p.y), self.PROJ_RADIUS, self.COLOR_ALIEN_PROJ)
            pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), self.PROJ_RADIUS, self.COLOR_ALIEN_PROJ)

        # Draw explosions
        for e in self.explosions:
            progress = e["timer"] / e["max_timer"]
            radius = int(e["radius"] * (1.2 - progress))
            alpha = int(255 * progress)
            color = (*self.COLOR_EXPLOSION, alpha)
            
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius, color)
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, color)
            self.screen.blit(temp_surf, (e["pos"][0] - radius, e["pos"][1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_ui.render(f"LIVES: {max(0, self.player_lives)}", True, self.COLOR_UI)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Game Over / Win message
        if self.game_over:
            if not self.aliens:
                msg = "YOU WIN"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_game_over.render(msg, True, self.COLOR_UI)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "aliens_remaining": len(self.aliens),
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment for human play
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "x11", "dummy", "fbcon", etc. as appropriate for your system

    env = GameEnv(render_mode="rgb_array")
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Minimalist Space Shooter")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Map keys to actions
    key_map = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    running = True
    while running:
        # Action defaults
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        # Pygame event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and done:
                obs, info = env.reset()
                done = False

        if not done:
            # Get key presses
            keys = pygame.key.get_pressed()
            for key, move_action in key_map.items():
                if keys[key]:
                    movement = move_action
            
            if keys[pygame.K_SPACE]:
                space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_held = 1

            action = [movement, space_held, shift_held]
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if done and running:
            # Game is over, wait for reset
            pass

    env.close()