import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move. Press space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade shooter. Destroy all descending aliens before they reach the bottom of the screen."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
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
        
        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_PLAYER_PROJ = (255, 255, 255)
        self.COLOR_ALIEN_PROJ = (255, 255, 0)
        self.COLOR_BOUNDARY = (200, 0, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_STAR = (100, 100, 120)

        # Fonts
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Game constants
        self.PLAYER_SPEED = 8
        self.PLAYER_FIRE_COOLDOWN = 6 # frames
        self.PLAYER_PROJ_SPEED = 12
        self.ALIEN_PROJ_SPEED = 5
        self.ALIEN_ROWS = 5
        self.ALIEN_COLS = 10
        self.TOTAL_ALIENS = self.ALIEN_ROWS * self.ALIEN_COLS
        self.MAX_STEPS = 10000
        self.BOUNDARY_Y = self.HEIGHT - 40
        
        # Initialize state variables
        self.player_pos = None
        self.aliens = None
        self.alien_data = None
        self.player_projectiles = None
        self.alien_projectiles = None
        self.particles = None
        self.stars = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.player_fire_timer = None
        self.prev_space_held = None
        self.alien_direction = None
        self.alien_descent_speed = None
        self.alien_fire_prob = None
        self.cleared_rows = None

        # self.np_random will be seeded by super().reset()
        self.reset()
        
        # Must be called after reset() to ensure all variables are initialized
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 60)
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.player_fire_timer = 0
        self.prev_space_held = False

        self._spawn_aliens()
        self.alien_direction = 1.5 # horizontal speed
        self.alien_descent_speed = 0.1
        self.alien_fire_prob = 0.005
        
        self.cleared_rows = set()

        if self.stars is None:
            self.stars = [
                (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
                for _ in range(100)
            ]

        return self._get_observation(), self._get_info()

    def _spawn_aliens(self):
        self.aliens = []
        self.alien_data = []
        alien_w, alien_h = 25, 20
        start_x = (self.WIDTH - self.ALIEN_COLS * (alien_w + 15)) / 2
        start_y = 50
        for r in range(self.ALIEN_ROWS):
            for c in range(self.ALIEN_COLS):
                x = start_x + c * (alien_w + 15)
                y = start_y + r * (alien_h + 15)
                rect = pygame.Rect(x, y, alien_w, alien_h)
                self.aliens.append(rect)
                self.alien_data.append({'original_row': r})
    
    def step(self, action):
        reward = 0.1  # Survival reward
        self.steps += 1
        
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Handle Player Input ---
        if movement == 3:  # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
        self.player_pos.x = np.clip(self.player_pos.x, 20, self.WIDTH - 20)
        
        # Fire on rising edge of space press, with cooldown
        if space_held and not self.prev_space_held and self.player_fire_timer == 0:
            # SFX: Player shoot
            self.player_projectiles.append(pygame.Rect(self.player_pos.x - 2, self.player_pos.y - 20, 4, 15))
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN
        self.prev_space_held = space_held

        # --- Update Game State ---
        self._update_projectiles()
        reward += self._update_aliens()
        self._update_particles()
        
        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 50 == 0:
            self.alien_descent_speed += 0.01
        if self.steps > 0 and self.steps % 100 == 0:
            self.alien_fire_prob = min(0.1, self.alien_fire_prob + 0.001)

        # --- Collision Detection ---
        collision_reward, missed_penalty = self._handle_collisions()
        reward += collision_reward
        reward += missed_penalty

        # --- Check Termination ---
        terminated = self.game_over
        if not self.aliens:
            if not self.game_won: # Award bonus only once
                reward += 100
                self.game_won = True
            terminated = True
            
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        # Final reward is sum of components
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_projectiles(self):
        for proj in self.player_projectiles:
            proj.y -= self.PLAYER_PROJ_SPEED
        for proj in self.alien_projectiles:
            proj.y += self.ALIEN_PROJ_SPEED

    def _update_aliens(self):
        if not self.aliens:
            return 0
        
        move_down = False
        min_x = min(a.left for a in self.aliens)
        max_x = max(a.right for a in self.aliens)
        
        if max_x + self.alien_direction > self.WIDTH or min_x + self.alien_direction < 0:
            self.alien_direction *= -1
            move_down = True
            
        for alien in self.aliens:
            alien.x += self.alien_direction
            if move_down:
                alien.y += 10
            alien.y += self.alien_descent_speed

            if alien.bottom > self.BOUNDARY_Y and not self.game_over:
                self.game_over = True
                # SFX: Game over
                return -100 # Terminal reward
        
        # Alien firing logic
        if self.np_random.random() < self.alien_fire_prob and self.aliens:
            # Find aliens at the bottom of each column
            bottom_aliens = {}
            for i, alien in enumerate(self.aliens):
                col = round((alien.centerx - 100) / 40) # Approximate column
                if col not in bottom_aliens or alien.bottom > self.aliens[bottom_aliens[col]].bottom:
                    bottom_aliens[col] = i
            
            if bottom_aliens:
                firing_alien_idx = self.np_random.choice(list(bottom_aliens.values()))
                firing_alien = self.aliens[firing_alien_idx]
                # SFX: Alien shoot
                self.alien_projectiles.append(pygame.Rect(firing_alien.centerx - 2, firing_alien.bottom, 4, 10))
        return 0

    def _handle_collisions(self):
        reward = 0
        missed_penalty = 0

        # Player projectiles vs Aliens
        for proj in self.player_projectiles[:]:
            for i, alien in reversed(list(enumerate(self.aliens))):
                if proj.colliderect(alien):
                    # SFX: Explosion
                    self._create_explosion(alien.center, self.COLOR_ALIEN)
                    
                    # Check for row clear bonus
                    alien_info = self.alien_data[i]
                    original_row = alien_info['original_row']
                    
                    self.aliens.pop(i)
                    self.alien_data.pop(i)
                    
                    is_row_clear = True
                    for other_alien_info in self.alien_data:
                        if other_alien_info['original_row'] == original_row:
                            is_row_clear = False
                            break
                    
                    if is_row_clear and original_row not in self.cleared_rows:
                        reward += 50
                        self.cleared_rows.add(original_row)
                        
                    self.player_projectiles.remove(proj)
                    reward += 10
                    self.score += 100
                    break
        
        # Remove off-screen projectiles
        for proj in self.player_projectiles[:]:
            if proj.bottom < 0:
                self.player_projectiles.remove(proj)
                missed_penalty -= 0.2
        
        self.alien_projectiles = [p for p in self.alien_projectiles if p.top < self.HEIGHT]
        
        # Alien projectiles vs Player (no damage, just visual)
        player_rect = pygame.Rect(self.player_pos.x - 15, self.player_pos.y - 10, 30, 20)
        for proj in self.alien_projectiles:
            if player_rect.colliderect(proj):
                # Could add a small negative reward or visual effect here
                pass

        return reward, missed_penalty

    def _create_explosion(self, pos, color):
        for _ in range(15):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            particle = {
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 31),
                'color': color,
                'size': self.np_random.integers(2, 6)
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for i in range(len(self.stars)):
            x, y, size = self.stars[i]
            y = (y + 0.5 * size) % self.HEIGHT
            self.stars[i] = (x, y, size)
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), size / 2)

    def _render_game(self):
        # Boundary line
        pygame.draw.line(self.screen, self.COLOR_BOUNDARY, (0, self.BOUNDARY_Y), (self.WIDTH, self.BOUNDARY_Y), 2)
        
        # Player
        player_points = [
            (self.player_pos.x, self.player_pos.y - 15),
            (self.player_pos.x - 15, self.player_pos.y + 10),
            (self.player_pos.x + 15, self.player_pos.y + 10)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, player_points)
        pygame.draw.aalines(self.screen, self.COLOR_PLAYER, True, player_points) # Anti-aliasing

        # Aliens
        for alien in self.aliens:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN, alien)

        # Projectiles
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, proj)
        for proj in self.alien_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN_PROJ, proj)
            
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            s = pygame.Surface((p['size'], p['size']), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p['pos'].x), int(p['pos'].y)))

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        aliens_left = len(self.aliens)
        aliens_text = self.font_main.render(f"ALIENS: {aliens_left}", True, self.COLOR_TEXT)
        self.screen.blit(aliens_text, (self.WIDTH - aliens_text.get_width() - 10, 10))

        if self.game_over or self.game_won:
            msg = "YOU LOSE"
            color = self.COLOR_ALIEN
            if self.game_won:
                msg = "YOU WIN!"
                color = self.COLOR_PLAYER
            
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "aliens_left": len(self.aliens),
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # Set to False to run a random agent
    MANUAL_PLAY = True 
    
    if MANUAL_PLAY:
        # Override auto_advance for human play to sync with display framerate
        env.auto_advance = False 
        
        # Pygame setup for display
        pygame.display.set_caption(env.game_description)
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        clock = pygame.time.Clock()
        
        obs, info = env.reset()
        done = False
        
        while not done:
            # Action defaults
            movement = 0 # none
            space = 0 # released
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space = 1

            action = [movement, space, 0] # shift is unused
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render to the display window
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Limit to 30 FPS for human play
            
            if done:
                print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
                pygame.time.wait(2000) # Pause before resetting
                obs, info = env.reset()
                done = False

    else: # Random Agent
        obs, info = env.reset()
        total_reward = 0
        for i in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if (i+1) % 100 == 0:
                print(f"Step {i+1}, Total Reward: {total_reward:.2f}, Info: {info}")
            if terminated or truncated:
                print(f"Episode finished after {i+1} steps. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0

    env.close()