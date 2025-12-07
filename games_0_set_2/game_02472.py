
# Generated: 2025-08-27T20:28:21.867738
# Source Brief: brief_02472.md
# Brief Index: 2472

        
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
        "Controls: ↑↓ to move your ship. Press Space to fire your weapon. Survive the onslaught!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade shooter. Survive for two minutes against waves of aliens while dodging their fire."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_TIME_SECONDS = 120
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS

        # Colors
        self.COLOR_BG = (10, 10, 25)
        self.COLOR_PLAYER_HEALTHY = (0, 255, 150)
        self.COLOR_PLAYER_HURT = (255, 255, 0)
        self.COLOR_PLAYER_CRITICAL = (255, 50, 50)
        self.COLOR_PLAYER_PROJ = (100, 255, 255)
        self.COLOR_ALIEN_PROJ = (255, 100, 255)
        self.COLOR_UI = (220, 220, 220)
        self.ALIEN_COLORS = {
            'small': (255, 70, 70),
            'medium': (255, 150, 50),
            'large': (200, 50, 255)
        }
        
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
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_health = 0
        self.player_fire_cooldown = 0
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        self.stars = []
        self.time_elapsed_seconds = 0
        self.alien_spawn_timer = 0
        self.current_alien_spawn_rate = 0
        self.current_alien_projectile_speed = 0
        self.reward_this_step = 0

        # Initialize state variables
        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed_seconds = 0
        
        # Player state
        self.player_pos = pygame.Vector2(self.WIDTH * 0.1, self.HEIGHT / 2)
        self.player_health = 3
        self.player_fire_cooldown = 0

        # Entity lists
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []

        # Difficulty scaling
        self.current_alien_spawn_rate = 2 * self.FPS # Initial spawn rate
        self.alien_spawn_timer = self.current_alien_spawn_rate
        self.current_alien_projectile_speed = 4

        # Initial aliens
        for _ in range(2):
            self._spawn_alien()
        
        # Background
        self.stars = [
            (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 2))
            for _ in range(150)
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.reward_this_step = 0
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            space_pressed = action[1] == 1  # Boolean
            
            # --- UPDATE GAME LOGIC ---
            self._update_player(movement, space_pressed)
            self._update_projectiles()
            self._update_aliens()
            self._update_particles()
            self._handle_collisions()
            self._update_difficulty()
            
            # Survival reward
            self.reward_this_step += 0.01

        # Update timers
        self.steps += 1
        self.time_elapsed_seconds = self.steps / self.FPS
        
        # Check termination conditions
        terminated = self._check_termination()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
        
    def _update_player(self, movement, space_pressed):
        player_speed = 7
        if movement == 1: # Up
            self.player_pos.y -= player_speed
        elif movement == 2: # Down
            self.player_pos.y += player_speed
        
        # Clamp player position
        self.player_pos.y = max(20, min(self.HEIGHT - 20, self.player_pos.y))

        # Handle firing
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
            
        if space_pressed and self.player_fire_cooldown == 0:
            # SFX: Player Laser Shot
            proj_pos = self.player_pos + pygame.Vector2(20, 0)
            self.player_projectiles.append({'pos': proj_pos, 'hit': False})
            self.player_fire_cooldown = 10 # 10-frame cooldown

    def _update_projectiles(self):
        player_proj_speed = 15
        
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj['pos'].x += player_proj_speed
            if proj['pos'].x > self.WIDTH:
                if not proj['hit']:
                    self.reward_this_step -= 0.2 # Miss penalty
                self.player_projectiles.remove(proj)

        # Alien projectiles
        for proj in self.alien_projectiles[:]:
            proj['pos'].x -= self.current_alien_projectile_speed
            if proj['pos'].x < 0:
                self.alien_projectiles.remove(proj)

    def _spawn_alien(self):
        alien_y = random.randint(20, self.HEIGHT - 20)
        roll = random.random()
        if roll < 0.6:
            alien_type = 'small'
            health = 1
            points = 1
        elif roll < 0.9:
            alien_type = 'medium'
            health = 2
            points = 2
        else:
            alien_type = 'large'
            health = 3
            points = 5
            
        self.aliens.append({
            'pos': pygame.Vector2(self.WIDTH + 30, alien_y),
            'type': alien_type,
            'health': health,
            'max_health': health,
            'points': points,
            'fire_timer': random.randint(1 * self.FPS, 3 * self.FPS),
            'speed': random.uniform(2.0, 4.0)
        })

    def _update_aliens(self):
        for alien in self.aliens[:]:
            alien['pos'].x -= alien['speed']
            if alien['pos'].x < -30:
                self.aliens.remove(alien)
                continue

            alien['fire_timer'] -= 1
            if alien['fire_timer'] <= 0:
                # SFX: Alien Laser Shot
                self.alien_projectiles.append({'pos': alien['pos'].copy()})
                alien['fire_timer'] = random.randint(int(1.5 * self.FPS), int(3.5 * self.FPS))

    def _update_difficulty(self):
        # Increase spawn rate every 15 seconds
        if self.steps > 0 and self.steps % (15 * self.FPS) == 0:
            self.current_alien_spawn_rate = max(20, self.current_alien_spawn_rate * 0.85)

        # Increase alien projectile speed every 30 seconds
        if self.steps > 0 and self.steps % (30 * self.FPS) == 0:
            self.current_alien_projectile_speed += 0.5
        
        self.alien_spawn_timer -= 1
        if self.alien_spawn_timer <= 0:
            self._spawn_alien()
            self.alien_spawn_timer = int(self.current_alien_spawn_rate)


    def _handle_collisions(self):
        # Player projectiles vs Aliens
        for proj in self.player_projectiles[:]:
            proj_rect = pygame.Rect(proj['pos'].x, proj['pos'].y, 15, 4)
            for alien in self.aliens[:]:
                alien_size = 15 + (alien['max_health'] * 5)
                alien_rect = pygame.Rect(alien['pos'].x - alien_size/2, alien['pos'].y - alien_size/2, alien_size, alien_size)
                if alien_rect.colliderect(proj_rect):
                    proj['hit'] = True
                    alien['health'] -= 1
                    # SFX: Alien Hit
                    self._create_explosion(proj['pos'], self.ALIEN_COLORS[alien['type']], 10)
                    if proj in self.player_projectiles:
                        self.player_projectiles.remove(proj)
                    
                    if alien['health'] <= 0:
                        # SFX: Alien Explosion
                        self.reward_this_step += alien['points']
                        self.score += alien['points']
                        self._create_explosion(alien['pos'], self.ALIEN_COLORS[alien['type']], 30)
                        self.aliens.remove(alien)
                    break 

        # Alien projectiles vs Player
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 10, 20, 20)
        for proj in self.alien_projectiles[:]:
            proj_rect = pygame.Rect(proj['pos'].x, proj['pos'].y, 15, 4)
            if player_rect.colliderect(proj_rect):
                # SFX: Player Hit/Explosion
                self.alien_projectiles.remove(proj)
                self.player_health -= 1
                self.reward_this_step -= 10
                self._create_explosion(self.player_pos, self.COLOR_PLAYER_CRITICAL, 40)
                if self.player_health <= 0:
                    self.game_over = True
                break

    def _check_termination(self):
        if self.game_over:
            return True
        
        if self.time_elapsed_seconds >= self.MAX_TIME_SECONDS:
            self.reward_this_step += 100 # Win bonus
            self.game_over = True
            return True
            
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
            
        return False

    def _create_explosion(self, position, color, num_particles):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = random.randint(15, 30)
            self.particles.append({'pos': position.copy(), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Stars
        for x, y, size in self.stars:
            self.screen.fill((200, 200, 255), (x, y, size, size))
        
        # Player
        if self.player_health > 0:
            if self.player_health == 3:
                player_color = self.COLOR_PLAYER_HEALTHY
            elif self.player_health == 2:
                player_color = self.COLOR_PLAYER_HURT
            else:
                player_color = self.COLOR_PLAYER_CRITICAL
            
            # Ship body (triangle)
            p1 = (self.player_pos.x + 15, self.player_pos.y)
            p2 = (self.player_pos.x - 15, self.player_pos.y - 10)
            p3 = (self.player_pos.x - 15, self.player_pos.y + 10)
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], player_color)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], player_color)
            
            # Engine glow
            glow_pos = (int(self.player_pos.x - 20), int(self.player_pos.y))
            glow_color = (255, 180, 50, 100)
            pygame.gfxdraw.filled_circle(self.screen, glow_pos[0], glow_pos[1], random.randint(6, 8), glow_color)

        # Player Projectiles
        for proj in self.player_projectiles:
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, (pos[0], pos[1]-2, 15, 4))
            pygame.gfxdraw.filled_circle(self.screen, pos[0]+7, pos[1], 4, (200, 255, 255, 100))

        # Alien Projectiles
        for proj in self.alien_projectiles:
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            pygame.draw.rect(self.screen, self.COLOR_ALIEN_PROJ, (pos[0]-15, pos[1]-2, 15, 4))
            pygame.gfxdraw.filled_circle(self.screen, pos[0]-7, pos[1], 4, (255, 180, 255, 100))

        # Aliens
        for alien in self.aliens:
            pos = (int(alien['pos'].x), int(alien['pos'].y))
            color = self.ALIEN_COLORS[alien['type']]
            size = 15 + (alien['max_health'] * 5)
            health_ratio = alien['health'] / alien['max_health']
            
            # Draw alien based on type
            if alien['type'] == 'small':
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(size/2), color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(size/2), color)
            elif alien['type'] == 'medium':
                rect = pygame.Rect(pos[0] - size/2, pos[1] - size/2, size, size)
                pygame.draw.rect(self.screen, color, rect)
            elif alien['type'] == 'large':
                points = []
                for i in range(6):
                    angle = math.pi / 3 * i
                    points.append((pos[0] + size/2 * math.cos(angle), pos[1] + size/2 * math.sin(angle)))
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            
            # Health bar
            if alien['health'] < alien['max_health']:
                bar_width = size
                bar_height = 5
                health_bar_width = int(bar_width * health_ratio)
                pygame.draw.rect(self.screen, (255,0,0), (pos[0] - bar_width/2, pos[1] - size/2 - 10, bar_width, bar_height))
                pygame.draw.rect(self.screen, (0,255,0), (pos[0] - bar_width/2, pos[1] - size/2 - 10, health_bar_width, bar_height))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            color = (*p['color'], alpha)
            size = int(5 * (p['lifespan'] / 30))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), max(0, size), color)


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, self.MAX_TIME_SECONDS - self.time_elapsed_seconds)
        minutes = int(time_left) // 60
        seconds = int(time_left) % 60
        timer_text = self.font_ui.render(f"TIME: {minutes:02}:{seconds:02}", True, self.COLOR_UI)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        # Player Health Icons
        heart_poly = [ (0,3), (3,0), (6,3), (3,7), (0,3) ]
        for i in range(self.player_health):
            points = [(p[0] + 15 + i*20, p[1] + 40) for p in heart_poly]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_CRITICAL)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER_CRITICAL)

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.player_health <= 0:
                end_text = "GAME OVER"
            else:
                end_text = "YOU SURVIVED!"
            
            text_surface = self.font_game_over.render(end_text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_seconds": self.time_elapsed_seconds,
            "player_health": self.player_health,
            "aliens_on_screen": len(self.aliens)
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
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Shooter")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        if keys[pygame.K_SPACE]:
            space_held = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}")

        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"--- Episode Finished ---")
            print(f"Final Score: {info['score']}")
            print(f"Time Survived: {info['time_seconds']:.2f}s")
            print("Resetting in 3 seconds...")
            pygame.time.wait(3000)
            env.reset()

        clock.tick(env.FPS)

    env.close()