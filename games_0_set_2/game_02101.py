
# Generated: 2025-08-28T03:41:55.509002
# Source Brief: brief_02101.md
# Brief Index: 2101

        
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
    An expert implementation of a Gymnasium environment for a fast-paced arcade racer.
    This environment prioritizes visual quality and engaging gameplay.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: ↑↓←→ to move. Hold Space for a speed boost. Dodge the aliens and reach the spaceship!"
    )

    # Short, user-facing description of the game
    game_description = (
        "Escape hordes of descending aliens and reach your spaceship in this fast-paced, top-down arcade game."
    )

    # Frames auto-advance at a consistent rate
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces as per requirements
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Visual and Game Design Constants
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_SPACESHIP = (200, 200, 220)
        self.COLOR_POWERUP = (255, 255, 255)
        self.ALIEN_COLORS = [(255, 80, 80), (80, 150, 255), (255, 255, 80)]
        self.UI_FONT = pygame.font.Font(None, 28)
        self.UI_COLOR = (220, 220, 220)

        # Game parameters
        self.PLAYER_SIZE = 16
        self.PLAYER_SPEED = 3.5
        self.BOOST_MULTIPLIER = 1.8
        self.ALIEN_SIZE = 18
        self.INITIAL_ALIEN_SPEED = 1.0
        self.MAX_ALIENS = 15
        self.POWERUP_SIZE = 10
        self.INVINCIBILITY_DURATION = 150 # in steps/frames
        self.MAX_STEPS = 10000
        self.INITIAL_LIVES = 5

        # State variables - initialized in reset()
        self.player_pos = None
        self.player_y_prev = None
        self.aliens = None
        self.powerups = None
        self.particles = None
        self.starfield = None
        self.steps = None
        self.score = None
        self.lives = None
        self.game_over = None
        self.alien_speed = None
        self.invincibility_timer = None

        # Initialize state
        self.reset()
        
        # Validate implementation after initialization
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.alien_speed = self.INITIAL_ALIEN_SPEED
        self.invincibility_timer = 0
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT - self.PLAYER_SIZE * 2], dtype=np.float32)
        self.player_y_prev = self.player_pos[1]

        self.aliens = []
        for _ in range(self.MAX_ALIENS):
            self._spawn_alien(is_initial=True)

        self.powerups = []
        self.particles = []

        if self.starfield is None:
            self.starfield = [
                (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
                for _ in range(150)
            ]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game Logic ---
        self.steps += 1
        reward = 0.1  # Base reward for survival

        # 1. Player Movement
        speed = self.PLAYER_SPEED * (self.BOOST_MULTIPLIER if space_held else 1.0)
        move_vector = np.array([0, 0], dtype=np.float32)
        if movement == 1: move_vector[1] -= 1  # Up
        elif movement == 2: move_vector[1] += 1  # Down
        elif movement == 3: move_vector[0] -= 1  # Left
        elif movement == 4: move_vector[0] += 1  # Right
        
        if np.linalg.norm(move_vector) > 0:
            move_vector /= np.linalg.norm(move_vector)
        
        self.player_pos += move_vector * speed
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE / 2, self.WIDTH - self.PLAYER_SIZE / 2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE / 2, self.HEIGHT - self.PLAYER_SIZE / 2)

        # Reward shaping for vertical movement
        if self.player_pos[1] > self.player_y_prev:
             reward -= 0.2 # Penalty for moving away from goal
        self.player_y_prev = self.player_pos[1]

        # Player particle trail
        if self.steps % (2 if space_held else 4) == 0 and np.linalg.norm(move_vector) > 0:
            self._create_particle(self.player_pos, count=1 if space_held else 0, p_type='trail')

        # 2. Update Timers
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1

        # 3. Update Aliens
        for alien in self.aliens:
            alien['pos'][1] += self.alien_speed
            if alien['pos'][1] > self.HEIGHT + self.ALIEN_SIZE:
                self.aliens.remove(alien)
                self._spawn_alien()
        
        # 4. Update Particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # 5. Spawning Logic
        if self.steps % 100 == 0:
            self.alien_speed += 0.1
        if self.steps % 300 == 0 and not self.powerups:
            self._spawn_powerup()

        # 6. Collision Detection
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE / 2, self.player_pos[1] - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Player vs Aliens
        if self.invincibility_timer == 0:
            for alien in self.aliens:
                alien_rect = pygame.Rect(alien['pos'][0] - self.ALIEN_SIZE / 2, alien['pos'][1] - self.ALIEN_SIZE / 2, self.ALIEN_SIZE, self.ALIEN_SIZE)
                if player_rect.colliderect(alien_rect):
                    # Sound effect placeholder: player_hit.wav
                    self.lives -= 1
                    reward -= 10
                    self.invincibility_timer = 60 # Brief invulnerability after hit
                    self._create_particle(self.player_pos, count=30, p_type='explosion')
                    self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT - self.PLAYER_SIZE * 2], dtype=np.float32)
                    break
        
        # Player vs Powerups
        for powerup in self.powerups:
            powerup_rect = pygame.Rect(powerup['pos'][0] - self.POWERUP_SIZE, powerup['pos'][1] - self.POWERUP_SIZE, self.POWERUP_SIZE*2, self.POWERUP_SIZE*2)
            if player_rect.colliderect(powerup_rect):
                # Sound effect placeholder: powerup_get.wav
                self.invincibility_timer = self.INVINCIBILITY_DURATION
                reward += 5
                self.score += 500
                self.powerups.remove(powerup)
                self._create_particle(self.player_pos, count=20, p_type='powerup_collect')
                break

        # 7. Check Termination Conditions
        terminated = False
        # Win condition: reach spaceship
        if self.player_pos[1] < 30:
            # Sound effect placeholder: win_game.wav
            reward += 100
            self.score += 10000
            terminated = True
            self.game_over = True
        
        # Lose condition: no lives left
        if self.lives <= 0:
            # Sound effect placeholder: game_over.wav
            reward -= 100 # Extra penalty on final hit
            terminated = True
            self.game_over = True

        # Max steps
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        self.score += 1 # Score for surviving a frame

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_alien(self, is_initial=False):
        y_pos = self.np_random.uniform(-self.HEIGHT, -self.ALIEN_SIZE) if not is_initial else self.np_random.uniform(0, self.HEIGHT * 0.7)
        self.aliens.append({
            'pos': np.array([self.np_random.uniform(self.ALIEN_SIZE, self.WIDTH - self.ALIEN_SIZE), y_pos], dtype=np.float32),
            'color': random.choice(self.ALIEN_COLORS)
        })
    
    def _spawn_powerup(self):
        self.powerups.append({
            'pos': np.array([self.np_random.uniform(50, self.WIDTH - 50), self.np_random.uniform(100, self.HEIGHT - 100)], dtype=np.float32)
        })

    def _create_particle(self, pos, count, p_type):
        for _ in range(count):
            if p_type == 'explosion':
                vel = np.array([self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)], dtype=np.float32)
                vel /= (np.linalg.norm(vel) + 1e-6)
                vel *= self.np_random.uniform(1, 5)
                life = self.np_random.integers(20, 40)
                color = random.choice([(255,100,0), (255,200,0), (255,255,255)])
            elif p_type == 'trail':
                vel = np.array([self.np_random.uniform(-0.2, 0.2), self.np_random.uniform(-0.2, 0.2)], dtype=np.float32)
                life = self.np_random.integers(10, 20)
                color = (int(self.COLOR_PLAYER[0]*0.7), int(self.COLOR_PLAYER[1]*0.7), int(self.COLOR_PLAYER[2]*0.7))
            elif p_type == 'powerup_collect':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 6)
                vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
                life = self.np_random.integers(30, 50)
                color = (200, 200, 255)
            else:
                continue

            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "invincibility_timer": self.invincibility_timer,
        }

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
        # Starfield
        for x, y, size in self.starfield:
            c = size * 40
            pygame.gfxdraw.pixel(self.screen, x, y, (c, c, c))

        # Spaceship (goal)
        ship_x, ship_y = self.WIDTH / 2, 20
        pygame.draw.polygon(self.screen, self.COLOR_SPACESHIP, [
            (ship_x, ship_y - 15), (ship_x - 20, ship_y + 10), (ship_x + 20, ship_y + 10)
        ])
        pygame.draw.rect(self.screen, (150, 150, 170), (ship_x - 10, ship_y + 10, 20, 5))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            size = 2 * (p['life'] / p['max_life'])
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))

        # Powerups
        for powerup in self.powerups:
            glow = math.sin(self.steps * 0.1) * 4 + 8
            self._draw_glowing_circle(self.screen, powerup['pos'], self.POWERUP_SIZE, self.COLOR_POWERUP, glow)

        # Aliens
        for alien in self.aliens:
            pos = (int(alien['pos'][0]), int(alien['pos'][1]))
            size = self.ALIEN_SIZE
            rect = (pos[0] - size/2, pos[1] - size/2, size, size)
            pygame.draw.rect(self.screen, alien['color'], rect, border_radius=3)

        # Player
        player_color = self.COLOR_PLAYER
        if self.invincibility_timer > 0:
            # Flashing effect when invincible
            if (self.invincibility_timer > 60 and self.steps % 10 < 5) or \
               (self.invincibility_timer <= 60 and self.steps % 4 < 2):
                 player_color = (255, 255, 255)
            
            # Glow effect when invincible
            glow_size = math.sin(self.steps * 0.2) * 3 + 10
            self._draw_glowing_circle(self.screen, self.player_pos, self.PLAYER_SIZE, (200, 255, 220), glow_size)

        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        player_rect = (px - self.PLAYER_SIZE / 2, py - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, player_color, player_rect, border_radius=3)

    def _render_ui(self):
        # Score
        score_text = self.UI_FONT.render(f"SCORE: {self.score}", True, self.UI_COLOR)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.UI_FONT.render("LIVES:", True, self.UI_COLOR)
        self.screen.blit(lives_text, (self.WIDTH - 150, 10))
        for i in range(self.lives):
            heart_pos_x = self.WIDTH - 70 + (i * 20)
            pygame.draw.polygon(self.screen, (255, 50, 50), [
                (heart_pos_x, 13), (heart_pos_x-7, 13), (heart_pos_x-7, 18), (heart_pos_x, 25), 
                (heart_pos_x+7, 18), (heart_pos_x+7, 13)
            ])

    def _draw_glowing_circle(self, surface, pos, radius, color, glow_size):
        pos_int = (int(pos[0]), int(pos[1]))
        for i in range(int(glow_size), 0, -1):
            alpha = int(100 * (1 - i / glow_size))
            pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], int(radius + i), (*color, alpha))
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], int(radius), color)
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], int(radius), color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode='rgb_array')
    
    # --- To actually see the game, we need a display window ---
    pygame.display.set_caption("Alien Escape")
    screen_display = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop for human play
    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(30) # Cap at 30 FPS for smooth playback
        
    env.close()
    print("Game Over!")
    print(f"Final Info: {info}")