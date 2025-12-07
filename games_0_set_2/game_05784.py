
# Generated: 2025-08-28T06:05:12.968985
# Source Brief: brief_05784.md
# Brief Index: 5784

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set a dummy video driver for headless operation
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move your ship. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend Earth from a descending alien horde in this retro-inspired arcade shooter. Clear all 3 stages to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.TOTAL_STAGES = 3
        self.INITIAL_LIVES = 3
        
        # Colors (Bright for interactive, dark for background)
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_INVINCIBLE = (150, 255, 150)
        self.COLOR_PROJECTILE = (255, 255, 200)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_EXPLOSION = [(255, 255, 0), (255, 150, 0), (200, 50, 0)]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_small = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_lives = 0
        self.player_invincibility_timer = 0
        self.projectiles = []
        self.aliens = []
        self.alien_block_rect = pygame.Rect(0,0,0,0)
        self.alien_lateral_direction = 1
        self.particles = []
        self.stars = []
        self.current_stage = 1
        self.stage_clear_timer = 0
        self.last_fire_step = 0
        self.fire_cooldown = 8 # steps
        
        # Initialize state
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 1
        self.stage_clear_timer = 0

        # Player state
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 40]
        self.player_lives = self.INITIAL_LIVES
        self.player_invincibility_timer = 120 # 4 seconds at 30fps
        
        # Entity lists
        self.projectiles = []
        self.particles = []
        
        # Generate game elements
        self._generate_stars()
        self._generate_aliens()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Handle stage clear transition
        if self.stage_clear_timer > 0:
            self.stage_clear_timer -= 1
            if self.stage_clear_timer == 0:
                self.current_stage += 1
                if self.current_stage > self.TOTAL_STAGES:
                    self.game_over = True # Game won
                else:
                    self._generate_aliens()
                    self.player_invincibility_timer = 60
            # During transition, only render, no game logic
            return self._get_observation(), 0.0, self.game_over, False, self._get_info()

        self.steps += 1
        reward += 0.01 # Small reward for surviving

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # Update player
        self._handle_player_input(movement, space_held)

        # Update game elements
        self._update_projectiles()
        self._update_aliens()
        self._update_particles()
        
        # Handle collisions and collect rewards
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # Check for stage clear
        if not self.aliens and self.stage_clear_timer == 0:
            self.stage_clear_timer = 90 # 3 seconds
            reward += 100
            self.score += 1000
            # SFX: Stage Clear Fanfare

        # Check for termination conditions
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held):
        # Movement
        player_speed = 8
        if movement == 3: # Left
            self.player_pos[0] -= player_speed
        elif movement == 4: # Right
            self.player_pos[0] += player_speed
        
        # Clamp player position
        self.player_pos[0] = max(20, min(self.WIDTH - 20, self.player_pos[0]))

        # Firing
        if space_held and (self.steps - self.last_fire_step) > self.fire_cooldown:
            # SFX: Player Shoot
            self.projectiles.append(pygame.Rect(self.player_pos[0] - 2, self.player_pos[1] - 20, 4, 15))
            self.last_fire_step = self.steps

        if self.player_invincibility_timer > 0:
            self.player_invincibility_timer -= 1

    def _update_projectiles(self):
        projectile_speed = 12
        for p in self.projectiles[:]:
            p.y -= projectile_speed
            if p.bottom < 0:
                self.projectiles.remove(p)

    def _update_aliens(self):
        descent_speed = 1.0 + (self.current_stage - 1) * 0.1
        lateral_speed = 0.5 + (self.current_stage - 1) * 0.2

        if not self.aliens: return

        # Update block rect
        self.alien_block_rect.left += lateral_speed * self.alien_lateral_direction
        self.alien_block_rect.top += descent_speed / 10.0 # Slow descent

        # Check for wall collision
        if self.alien_block_rect.right > self.WIDTH - 10 or self.alien_block_rect.left < 10:
            self.alien_lateral_direction *= -1
            self.alien_block_rect.top += 10 # Move down on wall hit

        # Update individual alien positions based on the block
        for alien in self.aliens:
            alien['pos'][0] += lateral_speed * self.alien_lateral_direction
            alien['pos'][1] += descent_speed / 10.0
            alien['rect'].topleft = alien['pos']

            # If any alien reaches the bottom, it's game over
            if alien['rect'].bottom > self.HEIGHT:
                self.game_over = True
                # SFX: Game Over
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0

        # Projectiles vs Aliens
        for p in self.projectiles[:]:
            for alien in self.aliens[:]:
                if p.colliderect(alien['rect']):
                    # SFX: Alien Explosion
                    self._create_explosion(alien['rect'].center, 20)
                    if p in self.projectiles: self.projectiles.remove(p)
                    self.aliens.remove(alien)
                    self.score += 10
                    reward += 1
                    self._recalculate_alien_block()
                    break

        # Aliens vs Player
        if self.player_invincibility_timer == 0:
            player_rect = pygame.Rect(self.player_pos[0] - 15, self.player_pos[1] - 10, 30, 20)
            for alien in self.aliens:
                if player_rect.colliderect(alien['rect']):
                    # SFX: Player Explosion
                    self._create_explosion(self.player_pos, 40)
                    self.player_lives -= 1
                    reward -= 10
                    if self.player_lives > 0:
                        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 40]
                        self.player_invincibility_timer = 120 # 4 seconds
                    break
        return reward
    
    def _check_termination(self):
        if self.game_over: return True
        
        if self.player_lives <= 0:
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
        elif self.current_stage > self.TOTAL_STAGES and not self.aliens:
             self.game_over = True # Win condition
        
        return self.game_over

    def _get_observation(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        self._render_stars()

        # Game elements
        self._render_aliens()
        self._render_projectiles()
        self._render_particles()
        self._render_player()
        
        # UI
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "stage": self.current_stage,
            "aliens_remaining": len(self.aliens),
        }

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            size = random.choice([1, 1, 1, 2, 3])
            brightness = random.randint(50, 150)
            self.stars.append(((x, y), size, (brightness, brightness, brightness)))
            
    def _generate_aliens(self):
        self.aliens = []
        rows, cols = 5, 10
        alien_size = 24
        spacing = 16
        start_x = (self.WIDTH - (cols * (alien_size + spacing))) / 2
        start_y = 50

        for row in range(rows):
            for col in range(cols):
                pos = [start_x + col * (alien_size + spacing), start_y + row * (alien_size + spacing)]
                rect = pygame.Rect(pos[0], pos[1], alien_size, alien_size)
                self.aliens.append({'pos': pos, 'rect': rect})
        
        self.alien_lateral_direction = 1
        self._recalculate_alien_block()

    def _recalculate_alien_block(self):
        if not self.aliens:
            self.alien_block_rect = pygame.Rect(0,0,0,0)
            return
        
        min_x = min(a['rect'].left for a in self.aliens)
        max_x = max(a['rect'].right for a in self.aliens)
        min_y = min(a['rect'].top for a in self.aliens)
        max_y = max(a['rect'].bottom for a in self.aliens)
        self.alien_block_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

    def _create_explosion(self, position, num_particles):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(15, 30)
            self.particles.append({
                'pos': list(position),
                'vel': vel,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': random.choice(self.COLOR_EXPLOSION)
            })

    def _render_stars(self):
        for pos, size, color in self.stars:
            if size == 1:
                self.screen.set_at(pos, color)
            else:
                pygame.draw.circle(self.screen, color, pos, size)

    def _render_player(self):
        if self.player_lives <= 0: return

        # Flash when invincible
        color = self.COLOR_PLAYER
        if self.player_invincibility_timer > 0 and (self.steps // 4) % 2 == 0:
            color = self.COLOR_PLAYER_INVINCIBLE

        p = self.player_pos
        points = [(p[0], p[1] - 15), (p[0] - 15, p[1] + 10), (p[0] + 15, p[1] + 10)]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_projectiles(self):
        for p in self.projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, p, border_radius=2)

    def _render_aliens(self):
        for alien in self.aliens:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN, alien['rect'], border_radius=4)
            # Add simple "eyes" for detail
            eye1_pos = (alien['rect'].centerx - 5, alien['rect'].centery - 3)
            eye2_pos = (alien['rect'].centerx + 5, alien['rect'].centery - 3)
            pygame.draw.circle(self.screen, self.COLOR_BG, eye1_pos, 2)
            pygame.draw.circle(self.screen, self.COLOR_BG, eye2_pos, 2)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = p['color']
            radius = int(3 * (p['lifespan'] / p['max_lifespan']))
            if radius > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*color, alpha))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*color, alpha))

    def _render_ui(self):
        # Score
        score_surf = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Lives
        lives_text_surf = self.font_small.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text_surf, (self.WIDTH - 200, 10))
        for i in range(self.player_lives):
            points = [
                (self.WIDTH - 110 + i * 30, 28),
                (self.WIDTH - 120 + i * 30, 38),
                (self.WIDTH - 100 + i * 30, 38)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)
        
        # Stage
        stage_surf = self.font_small.render(f"STAGE: {self.current_stage}", True, self.COLOR_TEXT)
        stage_rect = stage_surf.get_rect(center=(self.WIDTH/2, 25))
        self.screen.blit(stage_surf, stage_rect)

        # Game Over message
        if self.game_over:
            msg = "YOU WIN!" if self.current_stage > self.TOTAL_STAGES else "GAME OVER"
            text_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
        # Stage Clear message
        elif self.stage_clear_timer > 0:
            text_surf = self.font_large.render("STAGE CLEAR", True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

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

# Example usage to test the environment visually
if __name__ == '__main__':
    # Unset the dummy driver to allow rendering a window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Defender")
    clock = pygame.time.Clock()
    
    terminated = False
    
    print(env.user_guide)
    
    while not terminated:
        # Map keyboard keys to actions for human play
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Not used in this game
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        # Control the frame rate
        clock.tick(30)
        
    print(f"Game Over. Final Score: {info['score']}")
    env.close()