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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Hold Space to fire. Good luck, pilot!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro top-down space shooter. Survive waves of enemies and destroy them all to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # For auto_advance=True, this is the step rate

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (50, 255, 50, 40)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (255, 50, 50, 40)
        self.COLOR_PROJECTILE_PLAYER = (220, 220, 255)
        self.COLOR_PROJECTILE_ENEMY = (255, 200, 200)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_EXPLOSION = [(255, 100, 0), (255, 200, 0), (255, 255, 255)]

        # Gameplay Settings
        self.PLAYER_SPEED = 7
        self.PLAYER_FIRE_COOLDOWN = 5
        self.PLAYER_PROJECTILE_SPEED = 15
        self.PLAYER_LIVES = 3
        self.ENEMY_PROJECTILE_BASE_SPEED = 3
        self.ENEMY_FIRE_COOLDOWN_RANGE = (40, 80)
        self.MAX_ENEMIES_ON_SCREEN = 20
        self.TOTAL_ENEMIES_TO_SPAWN = 20
        self.ENEMY_SPAWN_INTERVAL = 50
        self.MAX_STEPS = 2000
        self.ENEMY_DIFFICULTY_INTERVAL = 500

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 20)
            self.font_title = pygame.font.Font(pygame.font.get_default_font(), 24)
        except IOError:
            self.font_ui = pygame.font.SysFont("monospace", 20)
            self.font_title = pygame.font.SysFont("monospace", 24)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_pos = None
        self.player_lives = None
        self.player_projectiles = None
        self.player_fire_cooldown_timer = None
        self.enemies = None
        self.enemy_projectiles = None
        self.particles = None
        self.enemies_killed = None
        self.enemies_spawned = None
        self.enemy_spawn_timer = None
        self.enemy_projectile_speed_bonus = None
        self.starfield = None
        self.np_random = None

        # --- Final Validation ---
        # This is called to ensure the environment conforms to the API.
        # It needs a fully initialized state, so we reset before calling it.
        # self.reset() # We reset inside the validation function now
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT - 50], dtype=np.float32)
        self.player_lives = self.PLAYER_LIVES
        self.player_projectiles = []
        self.player_fire_cooldown_timer = 0
        
        self.enemies = []
        self.enemy_projectiles = []
        self.particles = []
        
        self.enemies_killed = 0
        self.enemies_spawned = 0
        self.enemy_spawn_timer = self.ENEMY_SPAWN_INTERVAL
        self.enemy_projectile_speed_bonus = 0.0
        
        # Generate a static starfield for the episode
        self.starfield = [
            (
                self.np_random.integers(0, self.WIDTH),
                self.np_random.integers(0, self.HEIGHT),
                self.np_random.integers(1, 3)
            ) for _ in range(100)
        ]

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Action Handling ---
        movement = action[0]
        space_held = action[1] == 1
        # shift_held = action[2] == 1 # Unused in this design

        # Player Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED # Up
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED # Down
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED # Left
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED # Right
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)

        # Player Firing
        if self.player_fire_cooldown_timer > 0:
            self.player_fire_cooldown_timer -= 1
        
        if space_held and self.player_fire_cooldown_timer == 0:
            # sfx: player_shoot
            proj_pos = self.player_pos.copy()
            self.player_projectiles.append(proj_pos)
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN
        elif not space_held:
            reward -= 0.02 # Penalty for not firing

        # --- Game Logic Update ---
        self._update_difficulty()
        self._update_spawning()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        
        # --- Collision Detection ---
        reward += self._handle_collisions()

        # --- Termination Check ---
        terminated = False
        truncated = False
        if self.player_lives <= 0:
            terminated = True
            reward -= 50 # Lose penalty
            self.game_over = True
        elif self.enemies_killed >= self.TOTAL_ENEMIES_TO_SPAWN:
            terminated = True
            reward += 50 # Win bonus
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            terminated = True # For compatibility, also set terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.ENEMY_DIFFICULTY_INTERVAL == 0:
            self.enemy_projectile_speed_bonus += 0.05

    def _update_spawning(self):
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0 and self.enemies_spawned < self.TOTAL_ENEMIES_TO_SPAWN:
            self._spawn_enemy()
            self.enemies_spawned += 1
            self.enemy_spawn_timer = self.ENEMY_SPAWN_INTERVAL

    def _spawn_enemy(self):
        spawn_x = self.np_random.uniform(50, self.WIDTH - 50)
        pattern_type = self.np_random.choice(['sinusoidal', 'circular'])
        
        enemy = {
            'pos': np.array([spawn_x, -20], dtype=np.float32),
            'fire_cooldown': self.np_random.integers(*self.ENEMY_FIRE_COOLDOWN_RANGE),
            'pattern': pattern_type,
            'pattern_state': {
                'initial_y': self.np_random.uniform(40, 120),
                'amplitude': self.np_random.uniform(30, 80),
                'frequency': self.np_random.uniform(0.02, 0.05),
                'speed_x': self.np_random.uniform(-1.5, 1.5),
                'center_x': spawn_x,
            }
        }
        self.enemies.append(enemy)

    def _update_enemies(self):
        for enemy in self.enemies:
            # Movement
            state = enemy['pattern_state']
            if enemy['pattern'] == 'sinusoidal':
                enemy['pos'][0] += state['speed_x']
                enemy['pos'][1] = state['initial_y'] + math.sin(self.steps * state['frequency']) * state['amplitude']
                if not (0 < enemy['pos'][0] < self.WIDTH):
                    state['speed_x'] *= -1
            elif enemy['pattern'] == 'circular':
                angle = self.steps * state['frequency']
                enemy['pos'][0] = state['center_x'] + math.cos(angle) * state['amplitude']
                enemy['pos'][1] = state['initial_y'] + math.sin(angle) * state['amplitude']
            
            # Firing
            enemy['fire_cooldown'] -= 1
            if enemy['fire_cooldown'] <= 0 and self.player_lives > 0:
                # sfx: enemy_shoot
                direction = self.player_pos - enemy['pos']
                norm = np.linalg.norm(direction)
                if norm > 0:
                    velocity = (direction / norm) * (self.ENEMY_PROJECTILE_BASE_SPEED + self.enemy_projectile_speed_bonus)
                    self.enemy_projectiles.append({'pos': enemy['pos'].copy(), 'vel': velocity})
                enemy['fire_cooldown'] = self.np_random.integers(*self.ENEMY_FIRE_COOLDOWN_RANGE)

    def _update_projectiles(self):
        # Player projectiles
        for p in self.player_projectiles:
            p[1] -= self.PLAYER_PROJECTILE_SPEED
        self.player_projectiles = [p for p in self.player_projectiles if p[1] > -10]

        # Enemy projectiles
        for p in self.enemy_projectiles:
            p['pos'] += p['vel']
        self.enemy_projectiles = [p for p in self.enemy_projectiles if 0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT]

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Enemies
        new_player_projectiles = []
        for p_proj in self.player_projectiles:
            hit = False
            for enemy in self.enemies[:]:
                if np.linalg.norm(p_proj - enemy['pos']) < 15:
                    self._create_explosion(enemy['pos'], 30, self.COLOR_EXPLOSION)
                    self.enemies.remove(enemy)
                    self.score += 10
                    self.enemies_killed += 1
                    reward += 1
                    hit = True
                    break
            if not hit:
                new_player_projectiles.append(p_proj)
        self.player_projectiles = new_player_projectiles

        # Enemy projectiles vs Player
        for e_proj in self.enemy_projectiles[:]:
            if np.linalg.norm(e_proj['pos'] - self.player_pos) < 15 and self.player_lives > 0:
                self._create_explosion(self.player_pos, 50, self.COLOR_EXPLOSION)
                self.enemy_projectiles.remove(e_proj)
                self.player_lives -= 1
                self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT - 50], dtype=np.float32) # Respawn
                if self.player_lives <= 0:
                    self.game_over = True
                break

        # Player projectiles vs Enemy projectiles
        new_player_projectiles = []
        for p_proj in self.player_projectiles:
            hit = False
            for e_proj in self.enemy_projectiles[:]:
                if np.linalg.norm(p_proj - e_proj['pos']) < 10:
                    self._create_explosion(p_proj, 5, [self.COLOR_TEXT])
                    self.enemy_projectiles.remove(e_proj)
                    reward += 0.1
                    hit = True
                    break
            if not hit:
                new_player_projectiles.append(p_proj)
        self.player_projectiles = new_player_projectiles

        return reward

    def _create_explosion(self, pos, count, colors):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(10, 25),
                'color': random.choice(colors)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        if self.starfield:
            self._render_starfield()
        if self.particles:
            self._render_particles()
        if self.player_projectiles or self.enemy_projectiles:
            self._render_projectiles()
        if self.enemies:
            self._render_enemies()
        if self.player_lives > 0:
            self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_starfield(self):
        for x, y, size in self.starfield:
            color_val = 50 + size * 20
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (x, y), size)

    def _render_player(self):
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        
        # Glow effect
        glow_surface = pygame.Surface((60, 60), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_PLAYER_GLOW, (30, 30), 30)
        self.screen.blit(glow_surface, (px - 30, py - 30), special_flags=pygame.BLEND_RGBA_ADD)

        # Ship body
        points = [(px, py - 15), (px - 10, py + 10), (px + 10, py + 10)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Thruster - check for up action (1)
        # We don't have access to the action here, so we'll just flicker it
        if self.np_random.integers(0, 2) == 0:
             thrust_points = [(px - 5, py + 12), (px + 5, py + 12), (px, py + 18 + self.np_random.uniform(0, 5))]
             pygame.gfxdraw.aapolygon(self.screen, thrust_points, (255, 255, 100))
             pygame.gfxdraw.filled_polygon(self.screen, thrust_points, (255, 200, 0))


    def _render_enemies(self):
        for enemy in self.enemies:
            ex, ey = int(enemy['pos'][0]), int(enemy['pos'][1])
            
            # Glow effect
            glow_surface = pygame.Surface((50, 50), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, self.COLOR_ENEMY_GLOW, (25, 25), 25)
            self.screen.blit(glow_surface, (ex - 25, ey - 25), special_flags=pygame.BLEND_RGBA_ADD)

            # Ship body
            points = [(ex, ey + 12), (ex - 10, ey - 8), (ex + 10, ey - 8)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

    def _render_projectiles(self):
        for p in self.player_projectiles:
            start_pos = (int(p[0]), int(p[1]))
            end_pos = (int(p[0]), int(p[1]) + 8)
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE_PLAYER, start_pos, end_pos, 3)
        for p in self.enemy_projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_PROJECTILE_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, self.COLOR_PROJECTILE_ENEMY)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = max(1, int(p['lifespan'] / 5))
            pygame.draw.circle(self.screen, p['color'], pos, size)

    def _render_ui(self):
        # Lives
        for i in range(self.player_lives):
            heart_pos = (25 + i * 35, 25)
            points = [(heart_pos[0], heart_pos[1] - 5), (heart_pos[0] - 10, heart_pos[1] - 15), 
                      (heart_pos[0] - 10, heart_pos[1]), (heart_pos[0], heart_pos[1] + 10), 
                      (heart_pos[0] + 10, heart_pos[1]), (heart_pos[0] + 10, heart_pos[1] - 15)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 15))

        # Enemy Count
        enemies_left = self.TOTAL_ENEMIES_TO_SPAWN - self.enemies_killed
        enemy_text = self.font_ui.render(f"ENEMIES: {enemies_left}", True, self.COLOR_TEXT)
        self.screen.blit(enemy_text, (self.WIDTH / 2 - enemy_text.get_width() / 2, 15))

        if self.game_over:
            msg = "YOU WIN!" if self.player_lives > 0 else "GAME OVER"
            end_text = self.font_title.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - end_text.get_height()/2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_lives": self.player_lives,
            "enemies_killed": self.enemies_killed
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset, which initializes state and returns the first observation
        obs, info = self.reset()
        
        # Test the observation from reset
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape {obs.shape} is not {(self.HEIGHT, self.WIDTH, 3)}"
        assert obs.dtype == np.uint8, f"Obs dtype {obs.dtype} is not {np.uint8}"
        assert isinstance(info, dict)
        
        # Test step using the now-initialized environment
        test_action = self.action_space.sample()
        obs, reward, terminated, truncated, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play Example ---
    # This part requires a display to be available.
    # To run headlessly, comment out this section.
    try:
        # Unset the dummy video driver if we want to render
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
        
        pygame.display.init()
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Space Shooter Gym Environment")
        
        obs, info = env.reset()
        done = False
        
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

            # Render the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()

            # Handle window close
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            env.clock.tick(env.FPS)
            
        print(f"Game Over. Final Info: {info}")

    except pygame.error as e:
        print(f"Pygame display error: {e}")
        print("Skipping manual play example. The environment can still be used headlessly.")

    env.close()