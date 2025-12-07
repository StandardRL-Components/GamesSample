
# Generated: 2025-08-28T07:15:44.559918
# Source Brief: brief_03183.md
# Brief Index: 3183

        
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
        "Controls: ←→ to move. Hold space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down space shooter where you must destroy all invading aliens before they destroy you."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 10000
        self.TOTAL_ALIENS = 50

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_PLAYER_PROJ = (150, 255, 255)
        self.COLOR_ALIEN_PROJ = (255, 100, 255)
        self.COLOR_EXPLOSION = [(255, 200, 0), (255, 100, 0), (255, 255, 255)]
        self.COLOR_UI = (200, 200, 220)
        self.COLOR_HEART = (255, 0, 100)

        # Player settings
        self.PLAYER_SPEED = 8
        self.PLAYER_FIRE_COOLDOWN = 5 # frames
        self.PLAYER_WIDTH = 30
        self.PLAYER_HEIGHT = 20

        # Alien settings
        self.ALIEN_COLS = 10
        self.ALIEN_ROWS = self.TOTAL_ALIENS // self.ALIEN_COLS
        self.ALIEN_WIDTH = 24
        self.ALIEN_HEIGHT = 16
        self.ALIEN_H_SPACING = 40
        self.ALIEN_V_SPACING = 30
        self.ALIEN_DROP_AMOUNT = 10
        self.INITIAL_ALIEN_SPEED = 1.0
        self.INITIAL_ALIEN_FIRE_RATE = 0.002 # prob per alien per frame

        # Projectile settings
        self.PROJ_SPEED = 10
        self.PROJ_WIDTH = 4
        self.PROJ_HEIGHT = 12
        
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
        self.font_game_over = pygame.font.Font(None, 64)
        
        # Initialize state variables
        self.player_pos = None
        self.player_lives = None
        self.player_fire_cooldown_timer = None
        self.aliens = None
        self.alien_direction = None
        self.alien_speed = None
        self.alien_fire_rate = None
        self.aliens_destroyed_milestone = None
        self.projectiles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.aliens_destroyed = None
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        self.player_lives = 3
        self.player_fire_cooldown_timer = 0
        
        self.aliens = []
        start_x = (self.WIDTH - self.ALIEN_COLS * self.ALIEN_H_SPACING + self.ALIEN_H_SPACING - self.ALIEN_WIDTH) / 2
        start_y = 50
        for row in range(self.ALIEN_ROWS):
            for col in range(self.ALIEN_COLS):
                x = start_x + col * self.ALIEN_H_SPACING
                y = start_y + row * self.ALIEN_V_SPACING
                self.aliens.append({'rect': pygame.Rect(x, y, self.ALIEN_WIDTH, self.ALIEN_HEIGHT), 'alive': True})
        
        self.alien_direction = 1
        self.alien_speed = self.INITIAL_ALIEN_SPEED
        self.alien_fire_rate = self.INITIAL_ALIEN_FIRE_RATE
        self.aliens_destroyed = 0
        self.aliens_destroyed_milestone = 0

        self.projectiles = []
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Time penalty
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # Update game logic
        self._handle_input(movement, space_held)
        self._update_player()
        self._update_aliens()
        self._update_projectiles()
        self._update_particles()
        
        collision_rewards = self._handle_collisions()
        reward += collision_rewards
        
        self.steps += 1
        
        terminated = self._check_termination()
        
        if terminated:
            if self.win:
                reward += 100
            else:
                reward -= 100
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _handle_input(self, movement, space_held):
        # Player movement
        if movement == 3:  # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos.x += self.PLAYER_SPEED
        
        # Clamp player position
        self.player_pos.x = max(self.PLAYER_WIDTH / 2, min(self.WIDTH - self.PLAYER_WIDTH / 2, self.player_pos.x))

        # Player firing
        if space_held and self.player_fire_cooldown_timer <= 0:
            # SFX: Player shoot
            self.projectiles.append({
                'rect': pygame.Rect(self.player_pos.x - self.PROJ_WIDTH / 2, self.player_pos.y - self.PLAYER_HEIGHT, self.PROJ_WIDTH, self.PROJ_HEIGHT),
                'vel': -self.PROJ_SPEED,
                'type': 'player'
            })
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_player(self):
        if self.player_fire_cooldown_timer > 0:
            self.player_fire_cooldown_timer -= 1

    def _update_aliens(self):
        move_down = False
        for alien in self.aliens:
            if alien['alive']:
                if (self.alien_direction > 0 and alien['rect'].right > self.WIDTH) or \
                   (self.alien_direction < 0 and alien['rect'].left < 0):
                    move_down = True
                    break
        
        if move_down:
            self.alien_direction *= -1
            for alien in self.aliens:
                alien['rect'].y += self.ALIEN_DROP_AMOUNT
        else:
            for alien in self.aliens:
                alien['rect'].x += self.alien_direction * self.alien_speed
        
        # Alien firing
        living_aliens = [a for a in self.aliens if a['alive']]
        if living_aliens:
            fire_prob = self.alien_fire_rate * len(living_aliens)
            if self.np_random.random() < fire_prob:
                shooter = self.np_random.choice(living_aliens)
                # SFX: Alien shoot
                self.projectiles.append({
                    'rect': pygame.Rect(shooter['rect'].centerx - self.PROJ_WIDTH / 2, shooter['rect'].bottom, self.PROJ_WIDTH, self.PROJ_HEIGHT),
                    'vel': self.PROJ_SPEED / 2,
                    'type': 'alien'
                })

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj['rect'].y += proj['vel']
            if not self.screen.get_rect().colliderect(proj['rect']):
                self.projectiles.remove(proj)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_WIDTH / 2, self.player_pos.y - self.PLAYER_HEIGHT / 2, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)

        for proj in self.projectiles[:]:
            # Alien projectiles vs Player
            if proj['type'] == 'alien' and player_rect.colliderect(proj['rect']):
                self.player_lives -= 1
                self.projectiles.remove(proj)
                # SFX: Player hit
                self._create_explosion(self.player_pos, 20, self.COLOR_PLAYER)
                break # Only one hit per frame
            
            # Player projectiles vs Aliens
            if proj['type'] == 'player':
                for alien in self.aliens:
                    if alien['alive'] and alien['rect'].colliderect(proj['rect']):
                        alien['alive'] = False
                        if proj in self.projectiles:
                            self.projectiles.remove(proj)
                        
                        reward += 10.1 # +10 for kill, +0.1 for hit
                        self.score += 100
                        self.aliens_destroyed += 1
                        
                        # SFX: Alien explosion
                        self._create_explosion(pygame.Vector2(alien['rect'].center), 30, random.choice(self.COLOR_EXPLOSION))
                        
                        # Difficulty scaling
                        if self.aliens_destroyed // 10 > self.aliens_destroyed_milestone // 10:
                            self.alien_speed += 0.5
                        if self.aliens_destroyed // 5 > self.aliens_destroyed_milestone // 5:
                            self.alien_fire_rate += 0.001
                        
                        self.aliens_destroyed_milestone = self.aliens_destroyed
                        break
        return reward

    def _check_termination(self):
        if self.player_lives <= 0:
            return True
        
        if self.aliens_destroyed == self.TOTAL_ALIENS:
            self.win = True
            return True

        if self.steps >= self.MAX_STEPS:
            return True

        for alien in self.aliens:
            if alien['alive'] and alien['rect'].bottom >= self.player_pos.y - self.PLAYER_HEIGHT / 2:
                return True
        
        return False

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'life': life, 'color': color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            radius = int(p['life'] / 4)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), radius, (*p['color'], alpha))

        # Render aliens
        for alien in self.aliens:
            if alien['alive']:
                pygame.draw.rect(self.screen, self.COLOR_ALIEN, alien['rect'], border_radius=3)

        # Render projectiles
        for proj in self.projectiles:
            color = self.COLOR_PLAYER_PROJ if proj['type'] == 'player' else self.COLOR_ALIEN_PROJ
            pygame.draw.rect(self.screen, color, proj['rect'], border_radius=2)

        # Render player
        if self.player_lives > 0:
            p_w, p_h = self.PLAYER_WIDTH, self.PLAYER_HEIGHT
            p1 = (self.player_pos.x, self.player_pos.y - p_h / 2)
            p2 = (self.player_pos.x - p_w / 2, self.player_pos.y + p_h / 2)
            p3 = (self.player_pos.x + p_w / 2, self.player_pos.y + p_h / 2)
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1, p2, p3])
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

    def _render_ui(self):
        # Render score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))

        # Render lives
        heart_size = 12
        for i in range(self.player_lives):
            x = 20 + i * (heart_size * 2 + 5)
            y = 10 + heart_size
            p1 = (x, y - heart_size // 2)
            p2 = (x - heart_size, y - heart_size)
            p3 = (x + heart_size, y - heart_size)
            pygame.gfxdraw.filled_trigon(self.screen, int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), int(x), int(y+heart_size//2), self.COLOR_HEART)
            pygame.gfxdraw.aacircle(self.screen, int(x-heart_size//2), int(y-heart_size//2), int(heart_size//2), self.COLOR_HEART)
            pygame.gfxdraw.aacircle(self.screen, int(x+heart_size//2), int(y-heart_size//2), int(heart_size//2), self.COLOR_HEART)
            pygame.gfxdraw.filled_circle(self.screen, int(x-heart_size//2), int(y-heart_size//2), int(heart_size//2), self.COLOR_HEART)
            pygame.gfxdraw.filled_circle(self.screen, int(x+heart_size//2), int(y-heart_size//2), int(heart_size//2), self.COLOR_HEART)

        # Render Game Over/Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.COLOR_ALIEN
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "aliens_left": self.TOTAL_ALIENS - self.aliens_destroyed,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    # This block allows you to play the game directly
    # It will not be executed when the environment is used by an RL agent
    
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        # Map keys to action space
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # For human play, we need a display window
        if 'display' not in locals():
            display = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
            pygame.display.set_caption("Space Invaders")

        display.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Wait 3 seconds before resetting
            obs, info = env.reset()

        # Control the frame rate
        env.clock.tick(env.FPS)
        
    env.close()