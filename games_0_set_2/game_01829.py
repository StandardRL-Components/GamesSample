# Generated: 2025-08-28T02:49:46.221212
# Source Brief: brief_01829.md
# Brief Index: 1829

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string, corrected to match the shooter gameplay
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to fire."
    )

    # User-facing description of the game, corrected to match the shooter gameplay
    game_description = (
        "Defend Earth from a relentless alien invasion in this top-down arcade shooter. "
        "Evade enemy fire and destroy all aliens to win."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_MARGIN = 20

        # EXACT spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 72)

        # Colors
        self.COLOR_BG = (10, 0, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_ALIEN_GLOW = (255, 50, 50, 50)
        self.COLOR_PLAYER_PROJ = (255, 255, 0)
        self.COLOR_ALIEN_PROJ = (255, 100, 200)
        self.COLOR_EXPLOSION = [(255, 255, 100), (255, 150, 50), (200, 50, 0)]
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_STAR = (200, 200, 220)

        # Game constants
        self.MAX_STEPS = 10000
        self.TOTAL_ALIENS = 50
        self.INITIAL_LIVES = 3
        self.PLAYER_SPEED = 5
        self.PLAYER_FIRE_COOLDOWN = 6 # frames
        self.PLAYER_INVULNERABILITY = 60 # frames
        self.PLAYER_SIZE = 20
        self.ALIEN_SIZE = 22
        self.PROJECTILE_SPEED = 8

        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_lives = 0
        self.aliens_destroyed = 0
        self.player_rect = None
        self.player_cooldown = 0
        self.player_invulnerable_timer = 0
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        self.stars = []
        
        # Initialize state by calling reset
        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Commented out for final version, but useful for dev

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_lives = self.INITIAL_LIVES
        self.aliens_destroyed = 0
        
        player_start_pos = (self.WIDTH // 2, self.HEIGHT - self.PLAYER_SIZE * 2)
        self.player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.player_rect.center = player_start_pos
        self.player_cooldown = 0
        self.player_invulnerable_timer = 0

        self.aliens = self._spawn_aliens()
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        self.stars = [
            (
                self.np_random.integers(0, self.WIDTH),
                self.np_random.integers(0, self.HEIGHT),
                self.np_random.random() * 1.5,
            )
            for _ in range(150)
        ]
        
        return self._get_observation(), self._get_info()
    
    def _spawn_aliens(self):
        aliens = []
        rows = 5
        cols = self.TOTAL_ALIENS // rows
        for i in range(self.TOTAL_ALIENS):
            row = i // cols
            col = i % cols
            x = col * (self.ALIEN_SIZE + 20) + (self.WIDTH - cols * (self.ALIEN_SIZE + 20)) / 2
            y = row * (self.ALIEN_SIZE + 15) + 50
            alien_rect = pygame.Rect(x, y, self.ALIEN_SIZE, self.ALIEN_SIZE)
            aliens.append({
                "rect": alien_rect,
                "base_y": y,
                "phase": self.np_random.random() * 2 * math.pi
            })
        return aliens

    def step(self, action):
        reward = 0.1  # Survival reward per frame
        self.game_over = self.player_lives <= 0 or not self.aliens or self.steps >= self.MAX_STEPS
        if self.game_over:
            # If game is over, no actions should have an effect
            obs = self._get_observation()
            info = self._get_info()
            terminated = True
            return obs, 0, terminated, False, info

        # Unpack factorized action
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Decrement timers
        if self.player_cooldown > 0: self.player_cooldown -= 1
        if self.player_invulnerable_timer > 0: self.player_invulnerable_timer -= 1
        
        # Handle player actions
        self._handle_player_input(movement, space_held)
        if space_held and not self.aliens:
            reward -= 0.2 # Penalty for shooting with no targets
        
        # Update game state
        self._update_aliens()
        self._update_projectiles()
        self._update_particles()
        self._update_stars()
        
        # Handle collisions and calculate event-based rewards
        reward += self._handle_collisions()
        
        self.steps += 1
        
        # Check for termination conditions
        terminated = False
        truncated = False
        if not self.aliens: # Victory
            reward += 100
            terminated = True
            self.game_over = True
        elif self.player_lives <= 0: # Defeat
            reward += -100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS: # Timeout
            truncated = True # Use truncated for timeout
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held):
        # Movement
        if movement == 1: self.player_rect.y -= self.PLAYER_SPEED
        if movement == 2: self.player_rect.y += self.PLAYER_SPEED
        if movement == 3: self.player_rect.x -= self.PLAYER_SPEED
        if movement == 4: self.player_rect.x += self.PLAYER_SPEED

        # Engine particles when moving
        if movement != 0:
            self._create_particles(self.player_rect.midbottom, 3, [self.COLOR_EXPLOSION[1]], lifespan=10, speed_mult=0.5)

        # Clamp player to screen
        self.player_rect.left = max(self.WORLD_MARGIN, self.player_rect.left)
        self.player_rect.right = min(self.WIDTH - self.WORLD_MARGIN, self.player_rect.right)
        self.player_rect.top = max(self.WORLD_MARGIN, self.player_rect.top)
        self.player_rect.bottom = min(self.HEIGHT - self.WORLD_MARGIN, self.player_rect.bottom)

        # Firing
        if space_held and self.player_cooldown == 0:
            # sfx: player_shoot.wav
            proj_rect = pygame.Rect(0, 0, 4, 12)
            proj_rect.center = self.player_rect.midtop
            self.player_projectiles.append(proj_rect)
            self.player_cooldown = self.PLAYER_FIRE_COOLDOWN

    def _update_aliens(self):
        difficulty = self.aliens_destroyed // 10
        for alien in self.aliens:
            # Pattern 0: Static
            # Pattern 1: Sinusoidal horizontal
            if difficulty >= 1:
                alien['rect'].x += math.sin(alien['phase'] + self.steps * 0.05) * 2
            # Pattern 2: Sinusoidal vertical
            if difficulty >= 2:
                alien['rect'].y = alien['base_y'] + math.cos(alien['phase'] + self.steps * 0.03) * 30
            # Pattern 3: Diagonal drift
            if difficulty >= 3:
                alien['rect'].x += math.cos(alien['phase'] * 2) * 0.5
            # Pattern 4: Faster horizontal movement
            if difficulty >= 4:
                alien['rect'].x += math.sin(alien['phase'] + self.steps * 0.1) * 1.5

            # Firing logic
            fire_prob = 0.001 + difficulty * 0.001
            if self.np_random.random() < fire_prob:
                # sfx: alien_shoot.wav
                proj_rect = pygame.Rect(0, 0, 6, 6)
                proj_rect.center = alien['rect'].midbottom
                self.alien_projectiles.append(proj_rect)

    def _update_projectiles(self):
        self.player_projectiles = [p for p in self.player_projectiles if p.bottom > 0]
        for p in self.player_projectiles:
            p.y -= self.PROJECTILE_SPEED

        self.alien_projectiles = [p for p in self.alien_projectiles if p.top < self.HEIGHT]
        difficulty = self.aliens_destroyed // 10
        alien_proj_speed = self.PROJECTILE_SPEED * 0.5 + difficulty * 0.5
        for p in self.alien_projectiles:
            p.y += alien_proj_speed

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        aliens_to_remove = []
        projectiles_to_remove = []

        for i, proj in enumerate(self.player_projectiles):
            for j, alien in enumerate(self.aliens):
                if alien not in aliens_to_remove and proj.colliderect(alien['rect']):
                    aliens_to_remove.append(alien)
                    projectiles_to_remove.append(proj)
                    reward += 10
                    self.score += 100
                    self.aliens_destroyed += 1
                    # sfx: explosion.wav
                    self._create_particles(alien['rect'].center, 20, self.COLOR_EXPLOSION)
                    break # One projectile hits one alien
        
        self.aliens = [a for a in self.aliens if a not in aliens_to_remove]
        self.player_projectiles = [p for p in self.player_projectiles if p not in projectiles_to_remove]


        # Alien projectiles vs Player
        if self.player_invulnerable_timer == 0:
            hit_index = self.player_rect.collidelist(self.alien_projectiles)
            if hit_index != -1:
                self.alien_projectiles.pop(hit_index)
                self.player_lives -= 1
                reward -= 5
                self.player_invulnerable_timer = self.PLAYER_INVULNERABILITY
                # sfx: player_hit.wav
                self._create_particles(self.player_rect.center, 30, self.COLOR_EXPLOSION)
        
        return reward

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['radius'] -= p['decay']
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['radius'] > 0]

    def _update_stars(self):
        for i, (x, y, r) in enumerate(self.stars):
            y_new = y + r * 0.5 # Parallax scrolling
            if y_new > self.HEIGHT:
                y_new = 0
                x = self.np_random.integers(0, self.WIDTH)
            self.stars[i] = (x, y_new, r)

    def _create_particles(self, pos, count, colors, lifespan=30, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 * speed_mult + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.random() * 5 + 3,
                'color': random.choice(colors),
                'lifespan': self.np_random.integers(lifespan // 2, lifespan),
                'decay': 0.1 + self.np_random.random() * 0.2
            })
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw stars
        for x, y, r in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (int(x), int(y)), int(r))

        # Draw aliens
        for alien in self.aliens:
            pygame.gfxdraw.filled_circle(self.screen, alien['rect'].centerx, alien['rect'].centery, alien['rect'].width // 2, self.COLOR_ALIEN)
            pygame.gfxdraw.aacircle(self.screen, alien['rect'].centerx, alien['rect'].centery, alien['rect'].width // 2, self.COLOR_ALIEN)
            pygame.gfxdraw.aacircle(self.screen, alien['rect'].centerx, alien['rect'].centery, alien['rect'].width // 2 + 3, self.COLOR_ALIEN_GLOW)

        # Draw player projectiles
        for p in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, p)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, p.inflate(4, 4), 1)

        # Draw alien projectiles
        for p in self.alien_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, p.centerx, p.centery, p.width // 2, self.COLOR_ALIEN_PROJ)
            pygame.gfxdraw.aacircle(self.screen, p.centerx, p.centery, p.width // 2, self.COLOR_ALIEN_PROJ)

        # Draw player
        if self.player_lives > 0:
            is_invulnerable = self.player_invulnerable_timer > 0
            if not (is_invulnerable and (self.steps // 3) % 2 == 0): # Blink when invulnerable
                p = self.player_rect
                points = [(p.centerx, p.top), (p.left, p.bottom), (p.right, p.bottom)]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
                pygame.gfxdraw.aacircle(self.screen, p.centerx, p.centery, p.width, self.COLOR_PLAYER_GLOW)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 30))))
            try:
                color_with_alpha = p['color'] + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color_with_alpha)
            except (TypeError, ValueError):
                # Failsafe if color is somehow not a tuple
                # This can happen if random.choice picks an int from a color tuple
                # The primary fix is in _create_particles call, this is a guard
                pass


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Aliens remaining
        alien_text = self.font_ui.render(f"ALIENS: {len(self.aliens)}", True, self.COLOR_TEXT)
        self.screen.blit(alien_text, (self.WIDTH - alien_text.get_width() - 150, 10))

        # Lives
        for i in range(self.player_lives):
            p_size = self.PLAYER_SIZE * 0.7
            p_center = (self.WIDTH - (i + 1) * (p_size + 5), 10 + p_size / 2)
            points = [(p_center[0], p_center[1] - p_size/2), 
                      (p_center[0] - p_size/2, p_center[1] + p_size/2), 
                      (p_center[0] + p_size/2, p_center[1] + p_size/2)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        
        # Game Over / Win message
        if self.game_over:
            if not self.aliens and self.player_lives > 0:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_title.render(msg, True, self.COLOR_TEXT)
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

    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # For human play, we need a display
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Alien Invasion")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # Get player input from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        # Limit frame rate
        clock.tick(30)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before closing
            pygame.time.wait(3000)

    env.close()