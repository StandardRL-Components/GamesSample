
# Generated: 2025-08-27T16:25:21.050835
# Source Brief: brief_01216.md
# Brief Index: 1216

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move your ship. Hold Space to activate the mining beam on nearby asteroids."
    )

    game_description = (
        "Pilot a space miner through a dense asteroid field. Collect valuable ore by mining asteroids, "
        "but be careful to avoid collisions which will damage your ship. Collect 50 ore to win."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PLAYER_SPEED = 6
    PLAYER_RADIUS = 12
    MAX_HEALTH = 3
    WIN_SCORE = 50
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (15, 20, 40)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (50, 255, 150, 50)
    COLOR_ASTEROID = (120, 110, 100)
    COLOR_TEXT = (240, 240, 255)
    COLOR_HEALTH_HIGH = (0, 255, 0)
    COLOR_HEALTH_MID = (255, 255, 0)
    COLOR_HEALTH_LOW = (255, 0, 0)
    
    ORE_COLORS = {
        "gold": (255, 215, 0),
        "silver": (192, 192, 192),
        "copper": (184, 115, 51)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Set up headless Pygame
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        self.player_pos = None
        self.player_health = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.difficulty_multiplier = 1.0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_health = self.MAX_HEALTH
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.difficulty_multiplier = 1.0

        self.asteroids.clear()
        self.particles.clear()
        self.stars.clear()

        for _ in range(100):
            self.stars.append(
                (
                    self.np_random.integers(0, self.WIDTH),
                    self.np_random.integers(0, self.HEIGHT),
                    self.np_random.integers(1, 3),
                )
            )

        for _ in range(5):
            self._spawn_asteroid(on_screen=True)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action
        reward = -0.01  # Time penalty

        self._handle_input(movement)
        
        mining_reward = 0
        if space_held:
            mining_reward = self._handle_mining()
        else:
            self.active_mining_target = None
        
        reward += mining_reward

        self._update_game_state()
        
        collision_penalty = self._handle_collisions()
        reward += collision_penalty

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward = 100.0
            elif self.player_health <= 0:
                reward = -100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, movement):
        move_vec = np.array([0.0, 0.0])
        if movement == 1:  # Up
            move_vec = np.array([0, -1])
        elif movement == 2:  # Down
            move_vec = np.array([0, 1])
        elif movement == 3:  # Left
            move_vec = np.array([-1, 0])
        elif movement == 4:  # Right
            move_vec = np.array([1, 0])
        
        if np.linalg.norm(move_vec) > 0:
            self.player_pos += move_vec * self.PLAYER_SPEED
            # Add engine trail
            if self.steps % 2 == 0:
                trail_pos = self.player_pos - move_vec * 10
                trail_vel = -move_vec * self.np_random.uniform(0.5, 1.5)
                self._create_particle(trail_pos, trail_vel, self.COLOR_PLAYER, 5, 15)

        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)
        
    def _handle_mining(self):
        target_asteroid = None
        min_dist = float('inf')
        
        for asteroid in self.asteroids:
            if asteroid['ore'] > 0:
                dist = np.linalg.norm(self.player_pos - asteroid['pos'])
                if dist < min_dist:
                    min_dist = dist
                    target_asteroid = asteroid
        
        mining_range = 80
        reward = 0
        if target_asteroid and min_dist < mining_range:
            # // SFX: Mining beam hum
            target_asteroid['mining_progress'] = min(100, target_asteroid.get('mining_progress', 0) + 5)
            
            if target_asteroid['mining_progress'] >= 100:
                target_asteroid['mining_progress'] = 0
                ore_mined = 1
                target_asteroid['ore'] -= ore_mined
                self.score += ore_mined
                
                # Create ore particle
                p_vel = (self.player_pos - target_asteroid['pos']) / 30.0
                p_color = self.ORE_COLORS[target_asteroid['ore_type']]
                self._create_particle(target_asteroid['pos'].copy(), p_vel, p_color, 8, 40, is_ore=True)
                # // SFX: Ore collect ping
                
                reward += 0.1 * ore_mined

                if target_asteroid['ore'] <= 0:
                    # Asteroid depleted bonus
                    initial = target_asteroid['initial_ore']
                    if initial >= 6: reward += 1.0  # Large
                    elif initial >= 3: reward += 0.5 # Medium
                    else: reward += 0.2 # Small
                
                # Difficulty scaling
                if self.score > 0 and self.score % 10 == 0 and self.score != target_asteroid.get('last_difficulty_score', -1):
                    self.difficulty_multiplier *= 1.05
                    target_asteroid['last_difficulty_score'] = self.score

            self.active_mining_target = target_asteroid
        else:
            self.active_mining_target = None
            
        return reward

    def _update_game_state(self):
        # Update asteroids
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel'] * self.difficulty_multiplier
            asteroid['rotation'] = (asteroid['rotation'] + asteroid['rotation_speed']) % 360

        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if not p.get('is_ore', False):
                p['size'] *= 0.95
        
        # Cleanup
        self.asteroids = [a for a in self.asteroids if self._is_on_screen(a['pos'], a['radius'] + 20)]
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        
        # Spawn new asteroids
        if len(self.asteroids) < 10 + (self.difficulty_multiplier - 1) * 5:
            self._spawn_asteroid()

    def _handle_collisions(self):
        reward_penalty = 0
        asteroids_to_remove = []
        for i, asteroid in enumerate(self.asteroids):
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.PLAYER_RADIUS + asteroid['radius']:
                if i not in asteroids_to_remove:
                    # // SFX: Explosion sound
                    self.player_health -= 1
                    reward_penalty -= 1.0 # Immediate penalty
                    self._create_explosion(asteroid['pos'])
                    asteroids_to_remove.append(i)
        
        if asteroids_to_remove:
            self.asteroids = [a for i, a in enumerate(self.asteroids) if i not in asteroids_to_remove]
            
        return reward_penalty

    def _check_termination(self):
        if self.player_health <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _spawn_asteroid(self, on_screen=False):
        if on_screen:
            pos = self.np_random.uniform([0, 0], [self.WIDTH, self.HEIGHT])
        else:
            edge = self.np_random.integers(0, 4)
            if edge == 0:  # Top
                pos = np.array([self.np_random.uniform(0, self.WIDTH), -50.0])
            elif edge == 1:  # Bottom
                pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 50.0])
            elif edge == 2:  # Left
                pos = np.array([-50.0, self.np_random.uniform(0, self.HEIGHT)])
            else:  # Right
                pos = np.array([self.WIDTH + 50.0, self.np_random.uniform(0, self.HEIGHT)])

        center = np.array([self.WIDTH / 2, self.HEIGHT / 2])
        direction = center - pos + self.np_random.uniform(-50, 50, 2)
        vel = direction / np.linalg.norm(direction) * self.np_random.uniform(0.5, 1.5)

        ore_roll = self.np_random.random()
        if ore_roll < 0.5: # Small
            ore = self.np_random.integers(1, 3)
            radius = self.np_random.integers(15, 25)
            ore_type = "copper"
        elif ore_roll < 0.85: # Medium
            ore = self.np_random.integers(3, 6)
            radius = self.np_random.integers(25, 35)
            ore_type = "silver"
        else: # Large
            ore = self.np_random.integers(6, 10)
            radius = self.np_random.integers(35, 45)
            ore_type = "gold"

        shape_points = []
        num_points = self.np_random.integers(7, 12)
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            dist = radius * self.np_random.uniform(0.7, 1.1)
            shape_points.append((dist * math.cos(angle), dist * math.sin(angle)))

        self.asteroids.append({
            'pos': pos.astype(np.float32),
            'vel': vel.astype(np.float32),
            'ore': ore,
            'initial_ore': ore,
            'ore_type': ore_type,
            'radius': radius,
            'rotation': self.np_random.uniform(0, 360),
            'rotation_speed': self.np_random.uniform(-0.5, 0.5),
            'shape': shape_points,
            'mining_progress': 0,
        })

    def _create_particle(self, pos, vel, color, size, lifespan, is_ore=False):
        self.particles.append({
            'pos': pos.copy(),
            'vel': vel,
            'color': color,
            'size': size,
            'lifespan': lifespan,
            'is_ore': is_ore
        })

    def _create_explosion(self, pos):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            color = self.np_random.choice([(255, 69, 0), (255, 140, 0), (255, 255, 0)])
            size = self.np_random.uniform(5, 10)
            lifespan = self.np_random.integers(20, 40)
            self._create_particle(pos, vel, color, size, lifespan)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player_health}

    def _is_on_screen(self, pos, margin=0):
        return -margin < pos[0] < self.WIDTH + margin and -margin < pos[1] < self.HEIGHT + margin

    def _render_background(self):
        for x, y, size in self.stars:
            color_val = 100 + size * 20
            pygame.draw.rect(self.screen, (color_val, color_val, color_val), (x, y, size, size))

    def _render_game(self):
        # Render asteroids
        for asteroid in self.asteroids:
            self._draw_asteroid(asteroid)
        
        # Render mining beam
        if hasattr(self, 'active_mining_target') and self.active_mining_target:
            start_pos = tuple(self.player_pos.astype(int))
            end_pos = tuple(self.active_mining_target['pos'].astype(int))
            pygame.draw.aaline(self.screen, (100, 200, 255), start_pos, end_pos, 2)
            pygame.gfxdraw.filled_circle(self.screen, end_pos[0], end_pos[1], 5, (100, 200, 255, 150))
            
        # Render particles
        for p in self.particles:
            pos = p['pos'].astype(int)
            size = max(0, int(p['size']))
            if size > 0:
                alpha = int(255 * (p['lifespan'] / 40)) if p['lifespan'] < 40 else 255
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (pos[0] - size, pos[1] - size), special_flags=pygame.BLEND_RGBA_ADD)

        # Render player
        pos_int = self.player_pos.astype(int)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS + 5, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, (200, 255, 220))


    def _draw_asteroid(self, asteroid):
        pos = asteroid['pos']
        angle = math.radians(asteroid['rotation'])
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        
        # Rotated points for the main shape
        points = []
        for x, y in asteroid['shape']:
            rx = x * cos_a - y * sin_a + pos[0]
            ry = x * sin_a + y * cos_a + pos[1]
            points.append((int(rx), int(ry)))
        
        if len(points) > 2:
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)

        # Draw ore vein if ore is left
        if asteroid['ore'] > 0:
            vein_radius = int(asteroid['radius'] * 0.3 * (asteroid['ore'] / asteroid['initial_ore']))
            if vein_radius > 1:
                ore_color = self.ORE_COLORS[asteroid['ore_type']]
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), vein_radius, ore_color)
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), vein_radius, (255, 255, 255, 100))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"ORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Health Bar
        health_pct = self.player_health / self.MAX_HEALTH
        if health_pct > 0.6: health_color = self.COLOR_HEALTH_HIGH
        elif health_pct > 0.3: health_color = self.COLOR_HEALTH_MID
        else: health_color = self.COLOR_HEALTH_LOW
        
        bar_width = 100
        bar_height = 15
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 35, bar_width, bar_height))
        pygame.draw.rect(self.screen, health_color, (10, 35, int(bar_width * health_pct), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, 35, bar_width, bar_height), 1)

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                end_text = self.font_game_over.render("MISSION COMPLETE", True, self.COLOR_HEALTH_HIGH)
            else:
                end_text = self.font_game_over.render("SHIP DESTROYED", True, self.COLOR_HEALTH_LOW)
            
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def validate_implementation(self):
        print("✓ Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Requires pygame to be installed with display support
    # (i.e., don't run with SDL_VIDEODRIVER="dummy")
    
    # To run this, comment out the `os.environ['SDL_VIDEODRIVER'] = 'dummy'` line
    # in the __init__ method.
    
    class ManualPlayer:
        def __init__(self, env):
            self.env = env
            self.env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
            self.clock = pygame.time.Clock()

        def play(self):
            obs, info = self.env.reset()
            terminated = False
            
            while not terminated:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        terminated = True

                keys = pygame.key.get_pressed()
                
                movement = 0 # none
                if keys[pygame.K_UP]: movement = 1
                elif keys[pygame.K_DOWN]: movement = 2
                elif keys[pygame.K_LEFT]: movement = 3
                elif keys[pygame.K_RIGHT]: movement = 4
                
                space_held = 1 if keys[pygame.K_SPACE] else 0
                shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
                
                action = [movement, space_held, shift_held]
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Render the observation to the display
                frame = np.transpose(obs, (1, 0, 2))
                surf = pygame.surfarray.make_surface(frame)
                self.env.screen.blit(surf, (0, 0))
                pygame.display.flip()
                
                self.clock.tick(30)
                
                if terminated:
                    print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
                    pygame.time.wait(3000) # Pause for 3 seconds before closing

            self.env.close()

    # --- To run the manual player ---
    # 1. Make sure `os.environ['SDL_VIDEODRIVER'] = 'dummy'` is commented out.
    # 2. Uncomment the following lines:
    # env = GameEnv(render_mode="rgb_array")
    # player = ManualPlayer(env)
    # player.play()
    
    # --- To run a standard gym check ---
    # Make sure `os.environ['SDL_VIDEODRIVER'] = 'dummy'` is active.
    print("\nRunning standard Gym check...")
    from gymnasium.utils.env_checker import check_env
    env = GameEnv()
    check_env(env.unwrapped)
    print("✓ Gym check passed.")