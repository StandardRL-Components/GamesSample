# Generated: 2025-08-28T01:05:59.514147
# Source Brief: brief_04000.md
# Brief Index: 4000

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist top-down shooter. Destroy all invading aliens while dodging their projectiles to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG = (0, 0, 0)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_ALIEN = (255, 50, 50)
    COLOR_PLAYER_PROJ = (200, 200, 255)
    COLOR_ALIEN_PROJ = (255, 100, 100)
    COLOR_TEXT = (255, 255, 255)
    COLOR_PARTICLE = (255, 150, 50)
    
    # Screen
    WIDTH, HEIGHT = 640, 400

    # Game parameters
    PLAYER_SPEED = 5
    PLAYER_SIZE = 10
    PLAYER_FIRE_COOLDOWN = 6  # frames
    PLAYER_INVULNERABILITY = 90 # frames
    
    ALIEN_SIZE = 12
    ALIEN_ROWS = 5
    ALIEN_COLS = 10
    TOTAL_ALIENS = ALIEN_ROWS * ALIEN_COLS
    
    PROJ_SPEED = 8
    PROJ_SIZE = 2
    
    MAX_STEPS = 10000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_direction = None
        self.player_lives = None
        self.player_fire_cooldown = None
        self.player_invulnerable_timer = None
        self.muzzle_flash_timer = None
        
        self.aliens = None
        self.player_projectiles = None
        self.alien_projectiles = None
        self.particles = None
        
        self.alien_fire_rate = None
        self.alien_fire_cooldown = None
        
        self.steps = None
        self.score = None
        self.game_over = None
        self.victory = None
        
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        self.player_direction = pygame.math.Vector2(0, -1)
        self.player_lives = 3
        self.player_fire_cooldown = 0
        self.player_invulnerable_timer = 0
        self.muzzle_flash_timer = 0
        
        self.aliens = self._spawn_aliens()
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        
        self.alien_fire_rate = 0.2 # projectiles per second
        self.alien_fire_cooldown = 1.0 / self.alien_fire_rate if self.alien_fire_rate > 0 else float('inf')

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        return self._get_observation(), self._get_info()

    def _spawn_aliens(self):
        aliens = []
        x_spacing = 40
        y_spacing = 30
        x_offset = (self.WIDTH - (self.ALIEN_COLS - 1) * x_spacing) / 2
        y_offset = 50
        for row in range(self.ALIEN_ROWS):
            for col in range(self.ALIEN_COLS):
                aliens.append({
                    "pos": pygame.math.Vector2(x_offset + col * x_spacing, y_offset + row * y_spacing),
                    "initial_pos": pygame.math.Vector2(x_offset + col * x_spacing, y_offset + row * y_spacing),
                    "phase_offset": self.np_random.uniform(0, 2 * math.pi)
                })
        return aliens

    def step(self, action):
        reward = -0.01  # Small penalty for time passing

        if not self.game_over:
            # --- Update state based on action ---
            movement = action[0]
            space_held = action[1] == 1
            
            # --- Player Movement ---
            move_vec = pygame.math.Vector2(0, 0)
            if movement == 1: move_vec.y = -1
            elif movement == 2: move_vec.y = 1
            elif movement == 3: move_vec.x = -1
            elif movement == 4: move_vec.x = 1

            # Get distance to closest alien before moving
            old_dist_to_closest = self._get_dist_to_closest_alien()

            if move_vec.length() > 0:
                self.player_direction = move_vec.normalize()
                self.player_pos += self.player_direction * self.PLAYER_SPEED
                self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
                self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

            # --- Movement Reward ---
            new_dist_to_closest = self._get_dist_to_closest_alien()
            if new_dist_to_closest < old_dist_to_closest:
                reward += 0.1

            # --- Timers ---
            if self.player_fire_cooldown > 0: self.player_fire_cooldown -= 1
            if self.player_invulnerable_timer > 0: self.player_invulnerable_timer -= 1
            if self.muzzle_flash_timer > 0: self.muzzle_flash_timer -= 1
            self.alien_fire_cooldown -= 1 / 30.0 # Assuming 30fps

            # --- Player Shooting ---
            if space_held and self.player_fire_cooldown <= 0:
                # SFX placeholder: // Player Pew!
                proj_pos = self.player_pos + self.player_direction * (self.PLAYER_SIZE + 2)
                self.player_projectiles.append({"pos": proj_pos, "vel": self.player_direction * self.PROJ_SPEED})
                self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN
                self.muzzle_flash_timer = 2

            # --- Alien Logic ---
            self._update_aliens()
            self._update_alien_shooting()

            # --- Projectile Logic ---
            self._update_projectiles()

            # --- Collision Detection ---
            reward += self._handle_collisions()

            # --- Particle Logic ---
            self._update_particles()
        
        self.steps += 1
        
        # --- Check for termination ---
        terminated, term_reward = self._check_termination()
        reward += term_reward
        
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_dist_to_closest_alien(self):
        if not self.aliens:
            return 0
        return min(self.player_pos.distance_to(alien["pos"]) for alien in self.aliens)

    def _update_aliens(self):
        # Wave movement
        for alien in self.aliens:
            t = self.steps * 0.02
            alien['pos'].x = alien['initial_pos'].x + math.sin(t + alien['phase_offset']) * 20
            alien['pos'].y = alien['initial_pos'].y + (self.steps * 0.05)

    def _update_alien_shooting(self):
        # Increase fire rate over time
        self.alien_fire_rate = min(1.0, 0.2 + self.steps * 0.0001)

        if self.alien_fire_cooldown <= 0 and self.aliens:
            # SFX placeholder: // Alien Zap!
            firing_alien = self.np_random.choice(self.aliens)
            if (firing_alien['pos'] - self.player_pos).length() > 0:
                direction_to_player = (self.player_pos - firing_alien['pos']).normalize()
                proj_pos = firing_alien['pos'] + direction_to_player * (self.ALIEN_SIZE)
                self.alien_projectiles.append({"pos": proj_pos, "vel": direction_to_player * self.PROJ_SPEED * 0.75})
            
            fire_rate = self.alien_fire_rate if self.alien_fire_rate > 0 else float('inf')
            self.alien_fire_cooldown = 1.0 / fire_rate

    def _update_projectiles(self):
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj['pos'] += proj['vel']
            if not (0 <= proj['pos'].x <= self.WIDTH and 0 <= proj['pos'].y <= self.HEIGHT):
                self.player_projectiles.remove(proj)
        
        # Alien projectiles
        for proj in self.alien_projectiles[:]:
            proj['pos'] += proj['vel']
            if not (0 <= proj['pos'].x <= self.WIDTH and 0 <= proj['pos'].y <= self.HEIGHT):
                self.alien_projectiles.remove(proj)
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        # Player projectiles vs Aliens
        for proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                if proj['pos'].distance_to(alien['pos']) < (self.PROJ_SIZE + self.ALIEN_SIZE / 2):
                    # SFX placeholder: // Alien Explosion!
                    self.aliens.remove(alien)
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    self.score += 10
                    reward += 10
                    self._create_explosion(alien['pos'])
                    break
        
        # Alien projectiles vs Player
        if self.player_invulnerable_timer <= 0:
            player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE/2, self.player_pos.y - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
            for proj in self.alien_projectiles[:]:
                if player_rect.collidepoint(proj['pos']):
                    # SFX placeholder: // Player Hit!
                    self.alien_projectiles.remove(proj)
                    self.player_lives -= 1
                    self.player_invulnerable_timer = self.PLAYER_INVULNERABILITY
                    self._create_explosion(self.player_pos, self.COLOR_PLAYER)
                    break
        return reward

    def _create_explosion(self, pos, color=COLOR_PARTICLE, count=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'max_life': 30,
                'color': color
            })

    def _check_termination(self):
        reward = 0
        terminated = False
        if self.player_lives <= 0:
            self.game_over = True
            terminated = True
            reward = -100
        elif not self.aliens:
            self.game_over = True
            self.victory = True
            terminated = True
            reward = 100
        
        # Check if aliens moved off-screen
        for alien in self.aliens:
            if alien['pos'].y > self.HEIGHT:
                self.game_over = True
                terminated = True
                reward = -100 # Penalty for letting aliens pass
                break
        
        return terminated, reward

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            size = int(3 * (p['life'] / p['max_life']))
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), size)

        # Alien projectiles
        for proj in self.alien_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj['pos'].x), int(proj['pos'].y), self.PROJ_SIZE + 1, self.COLOR_ALIEN_PROJ)

        # Player projectiles
        for proj in self.player_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj['pos'].x), int(proj['pos'].y), self.PROJ_SIZE, self.COLOR_PLAYER_PROJ)
            pygame.gfxdraw.aacircle(self.screen, int(proj['pos'].x), int(proj['pos'].y), self.PROJ_SIZE, self.COLOR_PLAYER_PROJ)

        # Aliens
        for alien in self.aliens:
            pos = (int(alien['pos'].x), int(alien['pos'].y))
            size = self.ALIEN_SIZE
            pygame.draw.rect(self.screen, self.COLOR_ALIEN, (pos[0] - size / 2, pos[1] - size / 2, size, size))

        # Player
        is_invulnerable_flicker = self.player_invulnerable_timer > 0 and (self.steps // 3) % 2 == 0
        if not is_invulnerable_flicker:
            p1 = self.player_pos + self.player_direction * self.PLAYER_SIZE
            p2 = self.player_pos + self.player_direction.rotate(140) * self.PLAYER_SIZE * 0.8
            p3 = self.player_pos + self.player_direction.rotate(-140) * self.PLAYER_SIZE * 0.8
            points = [(int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (int(p3.x), int(p3.y))]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

            # Muzzle flash
            if self.muzzle_flash_timer > 0:
                flash_pos = self.player_pos + self.player_direction * (self.PLAYER_SIZE + 5)
                pygame.draw.circle(self.screen, (255, 255, 150), (int(flash_pos.x), int(flash_pos.y)), 5)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))

        # Lives
        for i in range(self.player_lives):
            points = [
                (20 + i * 25, 15),
                (10 + i * 25, 30),
                (30 + i * 25, 30)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Game Over / Victory Message
        if self.game_over:
            msg = "VICTORY" if self.victory else "GAME OVER"
            color = (100, 255, 100) if self.victory else (255, 50, 50)
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "aliens_remaining": len(self.aliens),
        }

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # Un-set the headless environment variable for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Shooter")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            # Wait for 'R' to restart
            pass

        clock.tick(30) # Run at 30 FPS

    env.close()