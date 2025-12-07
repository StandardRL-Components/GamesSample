
# Generated: 2025-08-28T02:01:57.129483
# Source Brief: brief_01577.md
# Brief Index: 1577

        
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
        "A vibrant, modern take on Space Invaders. Destroy waves of aliens, chain kills for a score multiplier, and survive the onslaught. Aliens get faster and more aggressive as you defeat them."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.COLOR_BG = (15, 15, 35)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_PROJ = (100, 200, 255)
        self.COLOR_ENEMY_PROJ = (255, 100, 100)
        self.ALIEN_COLORS = [(255, 100, 255), (255, 150, 100), (100, 255, 255), (255, 255, 100), (200, 100, 255)]
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_UI_ACCENT = (255, 200, 0)

        # Fonts
        self.font_main = pygame.font.Font(None, 36)
        self.font_big = pygame.font.Font(None, 72)
        self.font_multiplier = pygame.font.Font(None, 48)

        # Game parameters
        self.MAX_STEPS = 5000
        self.PLAYER_SPEED = 8
        self.PLAYER_FIRE_COOLDOWN = 6 # frames
        self.PLAYER_PROJECTILE_SPEED = 10
        self.ENEMY_PROJECTILE_SPEED = 4
        self.NUM_ALIENS_X = 10
        self.NUM_ALIENS_Y = 5
        self.TOTAL_ALIENS = self.NUM_ALIENS_X * self.NUM_ALIENS_Y

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_pos = 0
        self.lives = 0
        self.aliens = []
        self.alien_direction = 0
        self.alien_base_speed = 0
        self.alien_current_speed = 0
        self.alien_base_fire_prob = 0
        self.alien_current_fire_prob = 0
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        self.chain_multiplier = 1
        self.fire_cooldown = 0
        self.aliens_destroyed = 0
        self.last_action_was_fire = False

        self._create_starfield()
        
        # Initialize state
        self.reset()

        # Run validation check
        # self.validate_implementation() # Commented out for final submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        # Player state
        self.player_pos = self.WIDTH // 2
        self.lives = 3
        self.fire_cooldown = 0
        self.last_action_was_fire = False

        # Alien state
        self.alien_direction = 1
        self.alien_base_speed = 0.5
        self.alien_current_speed = self.alien_base_speed
        self.alien_base_fire_prob = 0.001
        self.alien_current_fire_prob = self.alien_base_fire_prob
        self.aliens_destroyed = 0
        self._setup_aliens()

        # Projectiles and effects
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        self.chain_multiplier = 1
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.02  # Small penalty for time passing

        if not self.game_over:
            # --- Handle Input ---
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            self._handle_input(movement, space_held)

            # --- Update Game Objects ---
            self._update_player()
            self._update_projectiles()
            self._update_aliens()
            self._update_particles()
            
            # --- Handle Collisions & Events ---
            collision_reward = self._handle_collisions()
            reward += collision_reward

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over: # First frame of termination
            self.game_over = True
            if self.win:
                reward += 100
            elif self.lives <= 0:
                reward -= 100
        
        # --- Return state ---
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        if movement == 3:  # Left
            self.player_pos -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos += self.PLAYER_SPEED
        
        self.player_pos = np.clip(self.player_pos, 30, self.WIDTH - 30)

        # Fire on key press (transition from not held to held)
        if space_held and not self.last_action_was_fire and self.fire_cooldown <= 0:
            # Sound: Player shoot
            self.player_projectiles.append(pygame.Rect(self.player_pos - 2, self.HEIGHT - 50, 4, 12))
            self.fire_cooldown = self.PLAYER_FIRE_COOLDOWN
        self.last_action_was_fire = space_held

    def _update_player(self):
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1

    def _update_projectiles(self):
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj.y -= self.PLAYER_PROJECTILE_SPEED
            if proj.bottom < 0:
                self.player_projectiles.remove(proj)
                self.chain_multiplier = 1 # Reset on miss

        # Alien projectiles
        for proj in self.alien_projectiles[:]:
            proj.y += self.ENEMY_PROJECTILE_SPEED
            if proj.top > self.HEIGHT:
                self.alien_projectiles.remove(proj)

    def _update_aliens(self):
        if not self.aliens:
            return

        move_down = False
        # Check for wall collision
        living_aliens = [a for a in self.aliens if a['alive']]
        if not living_aliens:
            return
            
        min_x = min(a['rect'].left for a in living_aliens)
        max_x = max(a['rect'].right for a in living_aliens)

        if (max_x >= self.WIDTH and self.alien_direction > 0) or \
           (min_x <= 0 and self.alien_direction < 0):
            self.alien_direction *= -1
            move_down = True

        # Move aliens
        for alien in self.aliens:
            if alien['alive']:
                if move_down:
                    alien['rect'].y += 15
                else:
                    alien['rect'].x += self.alien_current_speed * self.alien_direction
                
                # Check if alien reached bottom
                if alien['rect'].bottom >= self.HEIGHT - 50:
                    self.lives = 0 # Game over if any alien reaches the player's line
                    self.game_over = True
                
                # Alien firing
                if self.np_random.random() < self.alien_current_fire_prob:
                    # Sound: Alien shoot
                    self.alien_projectiles.append(pygame.Rect(alien['rect'].centerx - 2, alien['rect'].bottom, 5, 10))

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['radius'] -= 0.1
            if p['lifespan'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        # Player projectile vs Alien
        for proj in self.player_projectiles[:]:
            for alien in self.aliens:
                if alien['alive'] and proj.colliderect(alien['rect']):
                    # Sound: Alien explosion
                    alien['alive'] = False
                    self.player_projectiles.remove(proj)
                    
                    reward += 1 + self.chain_multiplier
                    self.score += 1 + self.chain_multiplier
                    self.chain_multiplier += 1
                    
                    self.aliens_destroyed += 1
                    self._create_explosion(alien['rect'].center, alien['color'], 20)

                    # Difficulty scaling
                    if self.aliens_destroyed % 10 == 0 and self.aliens_destroyed > 0:
                        self.alien_current_speed += 0.25
                        self.alien_current_fire_prob *= 1.5
                    
                    break # Projectile can only hit one alien

        # Alien projectile vs Player
        player_rect = pygame.Rect(self.player_pos - 15, self.HEIGHT - 40, 30, 20)
        for proj in self.alien_projectiles[:]:
            if player_rect.colliderect(proj):
                # Sound: Player hit
                self.alien_projectiles.remove(proj)
                self.lives -= 1
                self.chain_multiplier = 1 # Reset on hit
                self._create_explosion(player_rect.center, self.COLOR_PLAYER, 30)
                if self.lives <= 0:
                    self.game_over = True
                break
        return reward

    def _check_termination(self):
        self.steps += 1
        if self.aliens_destroyed == self.TOTAL_ALIENS:
            self.win = True
            return True
        if self.lives <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_aliens()
        self._render_player()
        self._render_projectiles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "chain_multiplier": self.chain_multiplier,
            "aliens_remaining": self.TOTAL_ALIENS - self.aliens_destroyed
        }

    # --- Helper and Rendering Methods ---

    def _create_starfield(self):
        self.stars = []
        for _ in range(150):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            size = random.uniform(0.5, 1.5)
            brightness = random.randint(50, 150)
            self.stars.append(((x, y), size, (brightness, brightness, brightness)))

    def _setup_aliens(self):
        self.aliens = []
        for y in range(self.NUM_ALIENS_Y):
            for x in range(self.NUM_ALIENS_X):
                alien_rect = pygame.Rect(
                    x * 45 + 80,
                    y * 40 + 50,
                    30, 20
                )
                self.aliens.append({
                    'rect': alien_rect,
                    'color': self.ALIEN_COLORS[y % len(self.ALIEN_COLORS)],
                    'alive': True
                })

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.random() * 3 + 2,
                'color': color,
                'lifespan': self.np_random.integers(15, 30)
            })

    def _render_background(self):
        for pos, size, color in self.stars:
            pygame.draw.circle(self.screen, color, pos, size)

    def _render_particles(self):
        for p in self.particles:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            radius_int = int(p['radius'])
            if radius_int > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius_int, p['color'])

    def _render_aliens(self):
        for alien in self.aliens:
            if alien['alive']:
                pygame.draw.rect(self.screen, alien['color'], alien['rect'], border_radius=4)
                # Eye
                eye_pos = (alien['rect'].centerx, alien['rect'].centery - 3)
                pygame.draw.circle(self.screen, self.COLOR_BG, eye_pos, 3)

    def _render_player(self):
        if self.lives > 0:
            p1 = (self.player_pos, self.HEIGHT - 40)
            p2 = (self.player_pos - 15, self.HEIGHT - 20)
            p3 = (self.player_pos + 15, self.HEIGHT - 20)
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1, p2, p3])
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

    def _render_projectiles(self):
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, proj, border_radius=3)
        for proj in self.alien_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_PROJ, proj, border_radius=3)

    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_surf, (10, 10))

        # Lives
        lives_surf = self.font_main.render(f"LIVES: {self.lives}", True, self.COLOR_WHITE)
        self.screen.blit(lives_surf, (self.WIDTH - lives_surf.get_width() - 10, 10))

        # Chain Multiplier
        if self.chain_multiplier > 1:
            mult_surf = self.font_multiplier.render(f"x{self.chain_multiplier}", True, self.COLOR_UI_ACCENT)
            self.screen.blit(mult_surf, (self.WIDTH // 2 - mult_surf.get_width() // 2, 10))

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.COLOR_ENEMY_PROJ
            msg_surf = self.font_big.render(msg, True, color)
            self.screen.blit(msg_surf, (self.WIDTH // 2 - msg_surf.get_width() // 2, self.HEIGHT // 2 - msg_surf.get_height() // 2))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    env.reset()
    
    # Use Pygame for human interaction
    import pygame
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Space Invaders")
    
    terminated = False
    running = True
    total_reward = 0
    
    # Set auto_advance to False for human play to sync with display loop
    env.auto_advance = False 
    
    while running:
        # Action defaults
        movement = 0 # no-op
        space_held = 0 # not pressed
        shift_held = 0 # not pressed
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()
                terminated = False
                total_reward = 0
        
        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space_held = 1
            
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            env.clock.tick(30) # Control frame rate for human play
        else:
            # Keep showing the last frame after game over
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    env.reset()
                    terminated = False
                    total_reward = 0

    env.close()