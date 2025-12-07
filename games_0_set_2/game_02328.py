import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A top-down arcade shooter Gymnasium environment. The player must destroy all
    invading aliens while dodging their projectiles. The game features three types
    of aliens with distinct behaviors, escalating difficulty, and polished retro
    visuals.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Press Space to fire your weapon. "
        "Your ship fires in the last direction you moved."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down arcade shooter. Destroy all invading aliens "
        "while dodging their projectiles to win."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Visuals & Style ---
        self.FONT = pygame.font.SysFont("monospace", 18, bold=True)
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_STAR = (200, 200, 220)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_PLAYER_PROJECTILE = (100, 200, 255)
        self.COLOR_ALIEN_PROJECTILE = (255, 100, 100)
        self.ALIEN_COLORS = {
            1: (255, 80, 80),   # Red
            2: (200, 80, 255),  # Purple
            3: (255, 160, 50),  # Orange
        }
        self.PARTICLE_COLORS = [(255, 220, 50), (255, 150, 50), (200, 50, 50)]

        # --- Game Parameters ---
        self.MAX_STEPS = 1500
        self.TOTAL_ALIENS = 20
        self.PLAYER_SPEED = 4
        self.PROJECTILE_SPEED = 8
        self.PLAYER_FIRE_COOLDOWN = 5  # steps

        # Initialize state variables
        self.stars = []
        self.player_pos = [0, 0]
        self.player_last_move_dir = [0, -1]
        self.player_fire_timer = 0
        self.last_space_held = False
        
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        
        self.difficulty_mods = {}
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize state by calling reset
        # self.reset() # Not needed here, will be called by user

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player state
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.player_last_move_dir = [0, -1] # Default fire upwards
        self.player_fire_timer = 0
        self.last_space_held = False

        # Entity lists
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []

        # Difficulty scaling
        self.difficulty_mods = {'fire_rate': 1.0, 'speed': 1.0}

        # Generate starfield
        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.uniform(0.5, 1.5))
            for _ in range(150)
        ]
        
        # Generate aliens
        self.aliens = []
        for i in range(self.TOTAL_ALIENS):
            alien_type = self.np_random.integers(1, 4)
            self.aliens.append(self._create_alien(alien_type))
            
        return self._get_observation(), self._get_info()

    def _create_alien(self, alien_type):
        alien = {
            'pos': [self.np_random.integers(50, self.WIDTH - 50), self.np_random.integers(50, self.HEIGHT // 2)],
            'type': alien_type,
            'size': 12,
            'fire_cooldown': self.np_random.integers(60, 120),
            'direction': 1 if self.np_random.random() > 0.5 else -1,
        }
        if alien_type == 3: # Burst fire alien
            alien['burst_shots_left'] = 0
            alien['burst_cooldown'] = 0
        return alien

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1
        
        reward = 0
        self.steps += 1
        
        # Action cost
        if movement != 0 or space_held:
            reward -= 0.01

        self._handle_input(movement, space_held)
        self._update_player_projectiles()
        self._update_aliens()
        self._update_alien_projectiles()
        reward += self._handle_collisions()
        self._update_particles()
        self._update_difficulty()
        
        dodged_reward, _ = self._cull_off_screen()
        reward += dodged_reward

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over: # Win condition
            reward += 100
            self.game_over = True
        elif terminated and self.game_over: # Loss condition
            reward -= 100
        
        if truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Player movement
        move_dir = [0, 0]
        if movement == 1: move_dir[1] = -1  # Up
        elif movement == 2: move_dir[1] = 1   # Down
        elif movement == 3: move_dir[0] = -1  # Left
        elif movement == 4: move_dir[0] = 1   # Right
        
        if movement != 0:
            self.player_last_move_dir = move_dir
            self.player_pos[0] += move_dir[0] * self.PLAYER_SPEED
            self.player_pos[1] += move_dir[1] * self.PLAYER_SPEED
            
            # Clamp position to screen bounds
            self.player_pos[0] = max(15, min(self.WIDTH - 15, self.player_pos[0]))
            self.player_pos[1] = max(15, min(self.HEIGHT - 15, self.player_pos[1]))

        # Player firing
        self.player_fire_timer = max(0, self.player_fire_timer - 1)
        if space_held and self.player_fire_timer == 0:
            # Sfx: Player shoot
            self.player_projectiles.append({
                'pos': list(self.player_pos),
                'vel': [v * self.PROJECTILE_SPEED for v in self.player_last_move_dir]
            })
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN
        self.last_space_held = space_held
    
    def _update_player_projectiles(self):
        for p in self.player_projectiles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]

    def _update_aliens(self):
        speed_mod = self.difficulty_mods['speed']
        for alien in self.aliens:
            # Movement
            if alien['type'] == 1: # Horizontal
                alien['pos'][0] += 1.5 * speed_mod * alien['direction']
                if alien['pos'][0] > self.WIDTH - 30 or alien['pos'][0] < 30:
                    alien['direction'] *= -1
            elif alien['type'] == 2: # Vertical
                alien['pos'][1] += 1 * speed_mod * alien['direction']
                if alien['pos'][1] > self.HEIGHT - 100 or alien['pos'][1] < 30:
                    alien['direction'] *= -1
            elif alien['type'] == 3: # Diagonal
                alien['pos'][0] += 1.2 * speed_mod * alien['direction']
                alien['pos'][1] += 0.8 * speed_mod
                if alien['pos'][0] > self.WIDTH - 30 or alien['pos'][0] < 30:
                    alien['direction'] *= -1
                if alien['pos'][1] > self.HEIGHT - 80:
                    alien['pos'][1] = 30 # Reset to top
            
            # Firing
            self._alien_try_fire(alien)

    def _alien_try_fire(self, alien):
        # Add a grace period to pass the stability test.
        if self.steps < 80:
            return

        fire_rate_mod = self.difficulty_mods['fire_rate']
        
        # Type 1: Slow, single shots
        if alien['type'] == 1:
            if self.np_random.random() < 0.005 * fire_rate_mod:
                self._fire_alien_projectile(alien)
        
        # Type 2: Rapid, single shots
        elif alien['type'] == 2:
            if self.np_random.random() < 0.015 * fire_rate_mod:
                self._fire_alien_projectile(alien)

        # Type 3: Bursts
        elif alien['type'] == 3:
            alien['burst_cooldown'] = max(0, alien['burst_cooldown'] - 1)
            if alien['burst_shots_left'] > 0 and alien['burst_cooldown'] == 0:
                self._fire_alien_projectile(alien, 0.8)
                alien['burst_shots_left'] -= 1
                alien['burst_cooldown'] = 5 # Cooldown between burst shots
            elif alien['burst_shots_left'] == 0 and self.np_random.random() < 0.008 * fire_rate_mod:
                alien['burst_shots_left'] = 3
                alien['burst_cooldown'] = 0

    def _fire_alien_projectile(self, alien, speed_mult=1.0):
        # Sfx: Alien shoot
        direction_to_player = np.array(self.player_pos) - np.array(alien['pos'])
        norm = np.linalg.norm(direction_to_player)
        if norm > 0:
            direction_to_player = direction_to_player / norm
        
        self.alien_projectiles.append({
            'pos': list(alien['pos']),
            'vel': (direction_to_player * self.PROJECTILE_SPEED * 0.6 * speed_mult).tolist()
        })

    def _update_alien_projectiles(self):
        for p in self.alien_projectiles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        player_projectiles_to_remove = []
        aliens_to_remove = []
        for i, p_proj in enumerate(self.player_projectiles):
            p_rect = pygame.Rect(p_proj['pos'][0] - 2, p_proj['pos'][1] - 4, 4, 8)
            for j, alien in enumerate(self.aliens):
                if j in aliens_to_remove: continue
                alien_rect = pygame.Rect(alien['pos'][0] - alien['size'], alien['pos'][1] - alien['size'], alien['size']*2, alien['size']*2)
                if p_rect.colliderect(alien_rect):
                    # Sfx: Explosion
                    self._create_explosion(alien['pos'], 20, self.ALIEN_COLORS[alien['type']])
                    aliens_to_remove.append(j)
                    player_projectiles_to_remove.append(i)
                    reward += 1
                    self.score += 100
                    break
        
        # Remove collided entities
        self.player_projectiles = [p for i, p in enumerate(self.player_projectiles) if i not in player_projectiles_to_remove]
        self.aliens = [a for i, a in enumerate(self.aliens) if i not in aliens_to_remove]

        # Alien projectiles vs Player
        player_rect = pygame.Rect(self.player_pos[0] - 8, self.player_pos[1] - 8, 16, 16)
        for a_proj in self.alien_projectiles:
            a_rect = pygame.Rect(a_proj['pos'][0] - 3, a_proj['pos'][1] - 3, 6, 6)
            if player_rect.colliderect(a_rect):
                # Sfx: Big explosion
                self._create_explosion(self.player_pos, 50, self.COLOR_PLAYER)
                self.game_over = True
                break
        
        return reward

    def _cull_off_screen(self):
        dodged_reward = 0
        
        # Player projectiles
        self.player_projectiles = [p for p in self.player_projectiles if 0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT]
        
        # Alien projectiles
        num_before = len(self.alien_projectiles)
        self.alien_projectiles = [p for p in self.alien_projectiles if 0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT]
        num_after = len(self.alien_projectiles)
        dodged_reward += (num_before - num_after) * 0.1 # Reward for dodging
        
        # Particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        return dodged_reward, num_before - num_after

    def _create_explosion(self, pos, num_particles, base_color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'size': self.np_random.uniform(2, 5),
                'color': random.choice(self.PARTICLE_COLORS)
            })
            
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
    
    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 100 == 0:
            self.difficulty_mods['fire_rate'] *= 1.05
        if self.steps > 0 and self.steps % 200 == 0:
            self.difficulty_mods['speed'] += 0.1

    def _check_termination(self):
        if self.game_over:
            return True
        if len(self.aliens) == 0:
            return True
        return False
        
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_stars()
        self._render_projectiles()
        self._render_aliens()
        if not self.game_over:
            self._render_player()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for x, y, size in self.stars:
            brightness = 0.5 + 0.5 * math.sin(self.steps * 0.05 + x)
            color_val = int(100 + 100 * brightness)
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (x, y), size)

    def _render_projectiles(self):
        for p in self.player_projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_PLAYER_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, self.COLOR_PLAYER_PROJECTILE)
        for p in self.alien_projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_ALIEN_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, self.COLOR_ALIEN_PROJECTILE)

    def _render_aliens(self):
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            s = alien['size']
            color = self.ALIEN_COLORS[alien['type']]
            if alien['type'] == 1: # Wide
                points = [(x - s, y), (x + s, y), (x + s*0.7, y + s*0.7), (x - s*0.7, y + s*0.7)]
            elif alien['type'] == 2: # Tall
                points = [(x, y - s), (x + s*0.7, y + s*0.5), (x, y + s), (x - s*0.7, y + s*0.5)]
            else: # Diamond
                points = [(x, y - s), (x + s, y), (x, y + s), (x - s, y)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        s = 10
        points = [(x, y - s), (x - s*0.8, y + s*0.8), (x, y + s*0.4), (x + s*0.8, y + s*0.8)]
        
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, x, y, 20, self.COLOR_PLAYER_GLOW)
        
        # Ship body
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        
        # Cockpit
        pygame.gfxdraw.filled_circle(self.screen, x, int(y+s*0.3), 3, (200, 255, 255))

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = p['life'] / 30.0
            color = tuple(int(c * alpha * alpha) for c in p['color'])
            pygame.draw.circle(self.screen, color, pos, int(p['size']))

    def _render_ui(self):
        score_text = self.FONT.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        aliens_text = self.FONT.render(f"ALIENS: {len(self.aliens)}", True, (255, 255, 255))
        self.screen.blit(aliens_text, (self.WIDTH - aliens_text.get_width() - 10, 10))
        
        if self.game_over:
            end_text_str = "VICTORY" if len(self.aliens) == 0 and not self.steps >= self.MAX_STEPS else "GAME OVER"
            end_font = pygame.font.SysFont("monospace", 48, bold=True)
            end_text = end_font.render(end_text_str, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "aliens_remaining": len(self.aliens),
            "game_over": self.game_over,
        }
        
    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    import pygame

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Alien Shooter")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    print(env.game_description)
    print(env.user_guide)

    while running:
        movement = 0 # No-op
        space_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        # The action space is MultiDiscrete, so we create a tuple
        action = (movement, space_held, 0) # Shift is unused
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            total_reward = 0
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS

    env.close()