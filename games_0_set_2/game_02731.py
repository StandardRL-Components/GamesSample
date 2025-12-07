
# Generated: 2025-08-28T05:46:22.205682
# Source Brief: brief_02731.md
# Brief Index: 2731

        
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


class Particle:
    """A single particle for effects like explosions."""
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = list(pos)
        self.vel = list(vel)
        self.radius = radius
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifespan -= 1
        self.radius -= 0.1
        self.vel[1] += 0.05 # a little gravity

    def draw(self, surface):
        if self.lifespan > 0 and self.radius > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            color_with_alpha = self.color + (alpha,)
            pygame.gfxdraw.filled_circle(
                surface, int(self.pos[0]), int(self.pos[1]), int(self.radius), color_with_alpha
            )
            pygame.gfxdraw.aacircle(
                surface, int(self.pos[0]), int(self.pos[1]), int(self.radius), color_with_alpha
            )


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move your ship. Press space to fire your laser."
    )

    game_description = (
        "A retro arcade shooter. Survive five waves of descending alien invaders. "
        "Destroy aliens for points and clear waves for a big bonus."
    )

    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 15, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_EXHAUST = (255, 100, 0)
    COLOR_ALIEN = (255, 50, 50)
    COLOR_PLAYER_PROJ = (200, 255, 255)
    COLOR_ALIEN_PROJ = (255, 100, 255)
    COLOR_EXPLOSION_1 = (255, 200, 0)
    COLOR_EXPLOSION_2 = (255, 100, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    
    # Game parameters
    PLAYER_SPEED = 5.0
    PLAYER_FIRE_COOLDOWN = 6 # frames
    PROJ_SPEED = 8.0
    ALIEN_SPEED = 0.3
    MAX_WAVES = 5
    MAX_STEPS = 3000 # Increased for 5 waves
    INITIAL_LIVES = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.stars = []
        self._create_stars()
        
        self.player_pos = [0, 0]
        self.player_lives = 0
        self.player_fire_timer = 0
        self.player_projectiles = []
        
        self.aliens = []
        self.alien_projectiles = []
        
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.current_wave = 0
        self.game_over = False
        self.win = False
        
        self.np_random = None
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 40]
        self.player_lives = self.INITIAL_LIVES
        self.player_fire_timer = 0
        
        self.player_projectiles.clear()
        self.aliens.clear()
        self.alien_projectiles.clear()
        self.particles.clear()
        
        self.current_wave = 1
        self._setup_wave(self.current_wave)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for time passing
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # 1. Handle Input & Player Actions
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Player movement
        old_player_pos = self.player_pos[:]
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], 15, self.SCREEN_WIDTH - 15)
        self.player_pos[1] = np.clip(self.player_pos[1], self.SCREEN_HEIGHT * 0.6, self.SCREEN_HEIGHT - 20)

        # Movement reward
        if len(self.aliens) > 0 and movement != 0:
            player_vec = np.array(self.player_pos)
            closest_alien = min(self.aliens, key=lambda a: np.linalg.norm(np.array(a['pos']) - player_vec))
            
            vec_to_alien = np.array(closest_alien['pos']) - np.array(old_player_pos)
            move_vec = np.array(self.player_pos) - np.array(old_player_pos)
            
            if np.linalg.norm(vec_to_alien) > 1 and np.linalg.norm(move_vec) > 1:
                cos_angle = np.dot(vec_to_alien, move_vec) / (np.linalg.norm(vec_to_alien) * np.linalg.norm(move_vec))
                if cos_angle > 0.5: # Moving generally towards alien
                    reward += 0.1

        # Player firing
        if self.player_fire_timer > 0: self.player_fire_timer -= 1
        if space_held and self.player_fire_timer == 0:
            self.player_projectiles.append(list(self.player_pos))
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN
            # Sound: Player shoot
            
            # Risky shot reward
            min_dist_to_alien_proj = float('inf')
            if len(self.alien_projectiles) > 0:
                player_vec = np.array(self.player_pos)
                for p in self.alien_projectiles:
                    dist = np.linalg.norm(player_vec - np.array(p))
                    min_dist_to_alien_proj = min(min_dist_to_alien_proj, dist)
            
            if min_dist_to_alien_proj < 150:
                reward += 2.0 # Risky shot bonus
            else:
                reward -= 1.0 # Safe shot penalty

        # 2. Update Game State
        self._update_projectiles()
        self._update_aliens()
        self._update_particles()
        
        # 3. Handle Collisions
        reward += self._handle_collisions()
        
        # 4. Check for Wave Completion
        if not self.aliens and not self.win:
            self.current_wave += 1
            reward += 100 # Wave clear bonus
            if self.current_wave > self.MAX_WAVES:
                self.win = True
                self.game_over = True
            else:
                self._setup_wave(self.current_wave)
        
        # 5. Check Termination Conditions
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.player_lives <= 0 and not self.win:
            reward -= 100 # Game over penalty
            self.game_over = True
            terminated = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _setup_wave(self, wave_num):
        num_cols = min(6 + wave_num, 10)
        num_rows = min(2 + wave_num, 5)
        
        x_spacing = self.SCREEN_WIDTH * 0.8 / num_cols
        y_spacing = 40
        
        start_x = (self.SCREEN_WIDTH - (num_cols - 1) * x_spacing) / 2
        start_y = 60

        shot_prob = 0.001 + (wave_num - 1) * 0.0015 # Per-alien, per-frame shot probability

        for row in range(num_rows):
            for col in range(num_cols):
                self.aliens.append({
                    'pos': [start_x + col * x_spacing, start_y + row * y_spacing],
                    'shot_prob': shot_prob,
                    'move_dir': 1,
                    'move_timer': 60
                })
    
    def _update_projectiles(self):
        # Player
        for proj in self.player_projectiles[:]:
            proj[1] -= self.PROJ_SPEED
            if proj[1] < 0:
                self.player_projectiles.remove(proj)
        # Alien
        for proj in self.alien_projectiles[:]:
            proj[1] += self.PROJ_SPEED / 2
            if proj[1] > self.SCREEN_HEIGHT:
                self.alien_projectiles.remove(proj)

    def _update_aliens(self):
        move_sideways = False
        for alien in self.aliens:
            if (alien['pos'][0] <= 20 and alien['move_dir'] == -1) or \
               (alien['pos'][0] >= self.SCREEN_WIDTH - 20 and alien['move_dir'] == 1):
                move_sideways = True
                break
        
        for alien in self.aliens:
            if move_sideways:
                alien['move_dir'] *= -1
                alien['pos'][1] += 10
            
            alien['pos'][0] += self.ALIEN_SPEED * alien['move_dir']
            
            if self.np_random.random() < alien['shot_prob']:
                self.alien_projectiles.append(list(alien['pos']))
                # Sound: Alien shoot

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0 or p.radius <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        for p_proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                dist = math.hypot(p_proj[0] - alien['pos'][0], p_proj[1] - alien['pos'][1])
                if dist < 15:
                    self._create_explosion(alien['pos'], self.COLOR_EXPLOSION_1, 20)
                    self.aliens.remove(alien)
                    if p_proj in self.player_projectiles: self.player_projectiles.remove(p_proj)
                    self.score += 100
                    reward += 10
                    # Sound: Alien explosion
                    break
        
        # Alien projectiles vs Player
        player_hitbox_radius = 10
        for a_proj in self.alien_projectiles[:]:
            dist = math.hypot(a_proj[0] - self.player_pos[0], a_proj[1] - self.player_pos[1])
            if dist < player_hitbox_radius:
                self.alien_projectiles.remove(a_proj)
                self.player_lives -= 1
                reward -= 5
                self._create_explosion(self.player_pos, self.COLOR_EXPLOSION_2, 30)
                # Sound: Player explosion
                if self.player_lives <= 0:
                    self.game_over = True
                break
        return reward
        
    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.random() * 3 + 3
            lifespan = self.np_random.integers(20, 40)
            self.particles.append(Particle(pos, vel, radius, color, lifespan))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "lives": self.player_lives,
            "aliens_remaining": len(self.aliens)
        }
        
    def _create_stars(self):
        self.stars = []
        for _ in range(150):
            x = random.randint(0, self.SCREEN_WIDTH)
            y = random.randint(0, self.SCREEN_HEIGHT)
            size = random.choice([1, 1, 1, 2])
            brightness = random.randint(50, 120)
            self.stars.append(((x, y), size, (brightness, brightness, brightness)))
    
    def _render_game(self):
        # Stars
        for pos, size, color in self.stars:
            pygame.draw.circle(self.screen, color, pos, size)
            
        # Player projectiles
        for proj in self.player_projectiles:
            p1 = (int(proj[0]), int(proj[1]))
            p2 = (int(proj[0]), int(proj[1] - 10))
            pygame.draw.line(self.screen, self.COLOR_PLAYER_PROJ, p1, p2, 3)
            
        # Alien projectiles
        for proj in self.alien_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj[0]), int(proj[1]), 4, self.COLOR_ALIEN_PROJ)
            pygame.gfxdraw.aacircle(self.screen, int(proj[0]), int(proj[1]), 4, self.COLOR_ALIEN_PROJ)

        # Aliens
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            points = [(x - 10, y + 5), (x + 10, y + 5), (x, y - 8)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ALIEN)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ALIEN)

        # Player ship
        if self.player_lives > 0:
            x, y = int(self.player_pos[0]), int(self.player_pos[1])
            # Main ship body
            ship_points = [(x, y - 12), (x - 10, y + 8), (x + 10, y + 8)]
            pygame.gfxdraw.aapolygon(self.screen, ship_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, ship_points, self.COLOR_PLAYER)
            # Exhaust flame
            flame_y = y + 8 + (self.steps % 4)
            flame_points = [(x - 5, y + 8), (x + 5, y + 8), (x, flame_y)]
            pygame.gfxdraw.aapolygon(self.screen, flame_points, self.COLOR_PLAYER_EXHAUST)
            pygame.gfxdraw.filled_polygon(self.screen, flame_points, self.COLOR_PLAYER_EXHAUST)

        # Particles
        for p in self.particles:
            p.draw(self.screen)
            
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Wave
        wave_text = self.font_ui.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))
        
        # Lives
        life_icon_base = [(0, -6), (-5, 4), (5, 4)]
        for i in range(self.player_lives):
            x_offset = 20 + i * 25
            y_offset = self.SCREEN_HEIGHT - 20
            points = [(p[0] + x_offset, p[1] + y_offset) for p in life_icon_base]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.COLOR_ALIEN
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Set a dummy action
    action = env.action_space.sample() 
    action[0] = 0 # No movement
    action[1] = 0 # No fire
    action[2] = 0 # No shift

    # Pygame setup for display
    pygame.display.set_caption("Arcade Shooter")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Manual Controls ---
        keys = pygame.key.get_pressed()
        
        # Reset action
        action[0] = 0 # Movement: none
        action[1] = 0 # Space: released
        action[2] = 0 # Shift: released

        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Match the intended FPS

    pygame.quit()