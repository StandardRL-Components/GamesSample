
# Generated: 2025-08-28T03:11:05.605549
# Source Brief: brief_01951.md
# Brief Index: 1951

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys (↑↓←→) to move your ship. Hold space to activate the mining beam."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship in a top-down asteroid field, mining valuable ore while dodging collisions."
    )

    # Should frames auto-advance or wait for user input?
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
        self.game_font = pygame.font.SysFont("monospace", 20, bold=True)
        self.game_over_font = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_ASTEROID = (150, 150, 150)
        self.COLOR_ORE = (255, 223, 0)
        self.COLOR_BEAM = (255, 255, 100)
        self.COLOR_EXPLOSION = [(255, 50, 50), (255, 150, 50), (200, 200, 200)]
        self.COLOR_UI = (255, 255, 255)

        # Game constants
        self.PLAYER_SPEED = 5
        self.PLAYER_RADIUS = 12
        self.MAX_HEALTH = 100
        self.WIN_SCORE = 50
        self.MAX_STEPS = 10000
        self.INITIAL_SPAWN_RATE = 0.1 # per second
        self.SPAWN_RATE_INCREASE = 0.001 # per second
        self.MINING_RANGE = 120
        self.MINING_WIDTH = 15
        self.MINING_RATE = 0.25 # ore per step
        self.LARGE_ASTEROID_THRESHOLD = 35

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_health = None
        self.player_angle = None
        self.last_move_action = 0
        self.score = None
        self.steps = None
        self.game_over = None
        self.asteroids = None
        self.particles = None
        self.stars = None
        self.current_spawn_rate = None
        
        self.reset()
        
        # Run validation
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0, 0], dtype=np.float32)
        self.player_health = self.MAX_HEALTH
        self.player_angle = -90.0
        self.last_move_action = 0
        
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self.asteroids = []
        self.particles = []
        
        self.current_spawn_rate = self.INITIAL_SPAWN_RATE

        # Generate a static starfield
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                'size': self.np_random.integers(1, 3),
                'layer': self.np_random.uniform(0.1, 0.5) # for parallax
            })

        for _ in range(5):
            self._spawn_asteroid(on_screen=True)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement, space_held, _ = action
        space_held = space_held == 1
        
        # Player movement
        move_vec = np.array([0, 0], dtype=np.float32)
        if movement != 0: self.last_move_action = movement
        if movement == 1: move_vec[1] = -1 # Up
        elif movement == 2: move_vec[1] = 1 # Down
        elif movement == 3: move_vec[0] = -1 # Left
        elif movement == 4: move_vec[0] = 1 # Right

        self.player_vel = move_vec * self.PLAYER_SPEED
        self.player_pos += self.player_vel
        
        # World wrapping
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT
        
        # Update player orientation for visuals
        if movement != 0:
            self.player_angle = math.degrees(math.atan2(move_vec[1], move_vec[0]))

        # --- Game Logic ---
        self.steps += 1
        ore_mined_this_step = 0
        
        # Update asteroids
        for ast in self.asteroids:
            ast['pos'] += ast['vel']
            ast['pos'][0] %= self.WIDTH
            ast['pos'][1] %= self.HEIGHT
            ast['angle'] = (ast['angle'] + ast['rot_speed']) % 360

        # Handle mining
        if space_held:
            mined_asteroids, mined_ore = self._handle_mining()
            ore_mined_this_step += mined_ore
            for ast in mined_asteroids:
                if ast['initial_size'] > self.LARGE_ASTEROID_THRESHOLD:
                    reward += 1 # Event-based reward for mining a large asteroid

        # Handle collisions
        collided = self._handle_collisions()
        if collided:
            # sound: player_hit.wav
            self.player_health -= 25
            self._spawn_explosion(self.player_pos, 30, self.COLOR_EXPLOSION)

        # Update particles
        self._update_particles()
        
        # Spawn new asteroids
        self.current_spawn_rate += self.SPAWN_RATE_INCREASE / 30.0 # Assuming 30fps
        if self.np_random.random() < self.current_spawn_rate / 30.0:
            self._spawn_asteroid()

        # --- Reward Calculation ---
        if ore_mined_this_step > 0:
            self.score += ore_mined_this_step
            reward += ore_mined_this_step * 0.1
        else:
            reward -= 0.02 # Penalty for inaction

        # --- Termination Check ---
        if self.player_health <= 0:
            terminated = True
            reward -= 100
            self.game_over = True
            # sound: game_over.wav
        elif self.score >= self.WIN_SCORE:
            self.score = self.WIN_SCORE # Cap score
            terminated = True
            reward += 100
            self.game_over = True
            # sound: victory.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_asteroid(self, on_screen=False):
        size = self.np_random.uniform(15, 50)
        
        if on_screen:
            # Avoid spawning on player
            while True:
                pos = np.array([self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)], dtype=np.float32)
                if np.linalg.norm(pos - self.player_pos) > size + self.PLAYER_RADIUS + 50:
                    break
        else:
            edge = self.np_random.integers(0, 4)
            if edge == 0: pos = np.array([-size, self.np_random.uniform(0, self.HEIGHT)], dtype=np.float32) # Left
            elif edge == 1: pos = np.array([self.WIDTH + size, self.np_random.uniform(0, self.HEIGHT)], dtype=np.float32) # Right
            elif edge == 2: pos = np.array([self.np_random.uniform(0, self.WIDTH), -size], dtype=np.float32) # Top
            else: pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + size], dtype=np.float32) # Bottom
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(0.5, 2.0)
        vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)

        # Generate a procedural shape
        num_verts = self.np_random.integers(6, 10)
        verts = []
        for i in range(num_verts):
            a = (i / num_verts) * 2 * math.pi
            r = size * self.np_random.uniform(0.7, 1.0)
            verts.append((math.cos(a) * r, math.sin(a) * r))

        self.asteroids.append({
            'pos': pos, 'vel': vel, 'size': size, 'initial_size': size,
            'ore': size, 'angle': self.np_random.uniform(0, 360),
            'rot_speed': self.np_random.uniform(-1.5, 1.5),
            'verts': verts
        })

    def _handle_mining(self):
        mined_asteroids = []
        total_ore_mined = 0
        
        # Define beam geometry
        beam_angle_rad = math.radians(self.player_angle)
        beam_dir = np.array([math.cos(beam_angle_rad), math.sin(beam_angle_rad)])
        
        asteroids_to_remove = []
        for i, ast in enumerate(self.asteroids):
            vec_to_ast = ast['pos'] - self.player_pos
            dist_to_ast = np.linalg.norm(vec_to_ast)
            
            if 0 < dist_to_ast < self.MINING_RANGE + ast['size']:
                # Check if asteroid is in the beam's cone
                if np.dot(beam_dir, vec_to_ast / dist_to_ast) > math.cos(math.atan(self.MINING_WIDTH / dist_to_ast)):
                    # sound: mining_beam.wav (loop)
                    ore_to_mine = min(ast['ore'], self.MINING_RATE)
                    ast['ore'] -= ore_to_mine
                    ast['size'] -= ore_to_mine * 0.8 # Shrink asteroid as it's mined
                    total_ore_mined += ore_to_mine
                    
                    if ast['ore'] <= 0:
                        asteroids_to_remove.append(i)
                        mined_asteroids.append(ast)
                        self._spawn_explosion(ast['pos'], int(ast['initial_size'] * 0.5), [self.COLOR_ASTEROID])
                    else:
                        # Create ore particles
                        for _ in range(2):
                            p_pos = ast['pos'] + self.np_random.uniform(-ast['size']/2, ast['size']/2, 2)
                            p_vel = (self.player_pos - p_pos) / 50.0 # Move towards player
                            p_vel += self.np_random.uniform(-0.5, 0.5, 2)
                            self.particles.append({'pos': p_pos, 'vel': p_vel, 'lifespan': 50, 'color': self.COLOR_ORE, 'size': 3})
        
        # Remove depleted asteroids
        for i in sorted(asteroids_to_remove, reverse=True):
            del self.asteroids[i]
            
        return mined_asteroids, total_ore_mined

    def _handle_collisions(self):
        collided = False
        for ast in self.asteroids:
            dist = np.linalg.norm(self.player_pos - ast['pos'])
            if dist < self.PLAYER_RADIUS + ast['size']:
                collided = True
                # Simple bounce physics
                ast['vel'] *= -1 
                break
        return collided

    def _spawn_explosion(self, pos, count, colors):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(), 'vel': vel,
                'lifespan': self.np_random.integers(20, 40),
                'color': random.choice(colors),
                'size': self.np_random.integers(2, 5)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['lifespan'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render parallax starfield
        for star in self.stars:
            px, py = self.player_pos
            sx, sy = star['pos']
            # Move stars opposite to player movement, scaled by layer
            render_x = (sx - px * star['layer']) % self.WIDTH
            render_y = (sy - py * star['layer']) % self.HEIGHT
            pygame.draw.circle(self.screen, (200, 200, 255), (int(render_x), int(render_y)), star['size'])
            
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 40.0))
            if alpha > 0:
                s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(s, p['color'] + (alpha,), (p['size'], p['size']), p['size'])
                self.screen.blit(s, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Render mining beam if active
        if self.action_space.sample()[1] == 1 and not self.game_over: # A bit of a hack to check if space is held
             beam_angle_rad = math.radians(self.player_angle)
             p1 = self.player_pos
             p2 = p1 + np.array([math.cos(beam_angle_rad - 0.05), math.sin(beam_angle_rad-0.05)]) * self.MINING_RANGE
             p3 = p1 + np.array([math.cos(beam_angle_rad + 0.05), math.sin(beam_angle_rad+0.05)]) * self.MINING_RANGE
             
             beam_points = [(int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1]))]
             pygame.gfxdraw.aapolygon(self.screen, beam_points, self.COLOR_BEAM)
             pygame.gfxdraw.filled_polygon(self.screen, beam_points, self.COLOR_BEAM + (50,))


        # Render asteroids
        for ast in self.asteroids:
            rotated_verts = []
            for vx, vy in ast['verts']:
                angle_rad = math.radians(ast['angle'])
                new_x = vx * math.cos(angle_rad) - vy * math.sin(angle_rad)
                new_y = vx * math.sin(angle_rad) + vy * math.cos(angle_rad)
                rotated_verts.append((int(ast['pos'][0] + new_x), int(ast['pos'][1] + new_y)))
            
            if len(rotated_verts) > 2:
                pygame.gfxdraw.aapolygon(self.screen, rotated_verts, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, rotated_verts, self.COLOR_ASTEROID)

        # Render player
        if not self.game_over:
            p_angle_rad = math.radians(self.player_angle)
            p_dir = np.array([math.cos(p_angle_rad), math.sin(p_angle_rad)])
            
            p1 = self.player_pos + p_dir * self.PLAYER_RADIUS
            p2 = self.player_pos + np.array([math.cos(p_angle_rad + 2.5), math.sin(p_angle_rad + 2.5)]) * self.PLAYER_RADIUS
            p3 = self.player_pos + np.array([math.cos(p_angle_rad - 2.5), math.sin(p_angle_rad - 2.5)]) * self.PLAYER_RADIUS
            
            player_points = [(int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1]))]
            pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)

            # Engine thrust effect
            if np.any(self.player_vel):
                for i in range(5):
                    offset = self.np_random.uniform(5, 15)
                    angle_offset = self.np_random.uniform(-0.3, 0.3)
                    pos = self.player_pos - p_dir * (self.PLAYER_RADIUS * 0.8 + offset)
                    size = self.np_random.uniform(1, 4)
                    color = random.choice([(255,200,0), (255,100,0)])
                    pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), int(size))


    def _render_ui(self):
        # Render Ore
        ore_text = self.game_font.render(f"ORE: {int(self.score)} / {self.WIN_SCORE}", True, self.COLOR_UI)
        self.screen.blit(ore_text, (10, 10))

        # Render Health
        health_text = self.game_font.render(f"HEALTH: {max(0, self.player_health)}%", True, self.COLOR_UI)
        health_rect = health_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(health_text, health_rect)

        # Health bar
        bar_width = 150
        bar_height = 15
        health_pct = max(0, self.player_health / self.MAX_HEALTH)
        fill_width = int(bar_width * health_pct)
        
        bg_rect = pygame.Rect(self.WIDTH - 10 - bar_width, 40, bar_width, bar_height)
        fill_rect = pygame.Rect(self.WIDTH - 10 - bar_width, 40, fill_width, bar_height)
        
        bar_color = (0, 255, 0) if health_pct > 0.5 else (255, 255, 0) if health_pct > 0.2 else (255, 0, 0)
        
        pygame.draw.rect(self.screen, (50, 50, 50), bg_rect)
        pygame.draw.rect(self.screen, bar_color, fill_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI, bg_rect, 1)

        # Game Over message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            if self.score >= self.WIN_SCORE:
                msg = "MISSION COMPLETE"
                color = self.COLOR_ORE
            else:
                msg = "GAME OVER"
                color = self.COLOR_EXPLOSION[0]
                
            over_text = self.game_over_font.render(msg, True, color)
            over_rect = over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(over_text, over_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "asteroids_count": len(self.asteroids)
        }

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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # For some reason pygame.display.set_mode must be called after env init
    # to show the window.
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption(env.game_description)
    
    obs, info = env.reset()
    terminated = False
    
    print(env.user_guide)
    
    while not terminated:
        # Action defaults
        movement = 0 # no-op
        space_held = 0 # released
        shift_held = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Control FPS

    print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
    env.close()