
# Generated: 2025-08-28T04:44:02.844971
# Source Brief: brief_05344.md
# Brief Index: 5344

        
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
        "Controls: Arrow keys to move your ship. Hold Space near an asteroid to mine it for ore."
    )

    game_description = (
        "Pilot a spaceship to mine asteroids for valuable ore. Avoid collisions and collect 100 ore to win."
    )

    auto_advance = True

    # --- Constants ---
    # Game parameters
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    WIN_SCORE = 100
    MAX_COLLISIONS = 5
    MAX_STEPS = 1800  # 60 seconds at 30 FPS

    # Player settings
    PLAYER_ACCELERATION = 0.5
    PLAYER_FRICTION = 0.96
    PLAYER_MAX_SPEED = 6
    PLAYER_RADIUS = 10
    PLAYER_HEALTH = 5

    # Asteroid settings
    ASTEROID_COUNT = 15
    ASTEROID_MIN_RADIUS = 15
    ASTEROID_MAX_RADIUS = 40
    ASTEROID_MIN_VERTICES = 5
    ASTEROID_MAX_VERTICES = 12
    ASTEROID_ORE_DENSITY = 2.5

    # Mining settings
    MINING_RANGE = 60
    MINING_RATE = 0.5

    # Reward settings
    REWARD_PER_ORE = 0.1
    REWARD_DEPLETE_ASTEROID = 1.0
    REWARD_WIN = 100.0
    REWARD_LOSE = -100.0
    PENALTY_COLLISION = -20.0
    PENALTY_PROXIMITY = -0.5
    PROXIMITY_THRESHOLD = 30

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (0, 255, 150, 50)
    COLOR_ASTEROID = (120, 120, 130)
    COLOR_ASTEROID_OUTLINE = (160, 160, 170)
    COLOR_ORE = (255, 220, 0)
    COLOR_EXPLOSION = [(255, 50, 50), (255, 150, 0), (255, 255, 255)]
    COLOR_THRUSTER = [(100, 150, 255), (200, 220, 255)]
    COLOR_MINING_BEAM = (50, 255, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_BAR_HEALTH = (0, 200, 100)
    COLOR_BAR_HEALTH_WARN = (255, 200, 0)
    COLOR_BAR_HEALTH_DANGER = (220, 50, 50)
    COLOR_BAR_BG = (50, 50, 70)


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = 0
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.mining_target = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.PLAYER_HEALTH

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)

        self.asteroids = self._create_asteroids()
        self.stars = self._create_stars()
        self.particles = []
        self.mining_target = None

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action
        reward = 0
        self.mining_target = None
        
        if not self.game_over:
            # --- Update Game Logic ---
            self._handle_input(movement)
            self._update_player()
            
            ore_mined, asteroid_depleted = self._update_mining(space_held == 1)
            collision_detected = self._handle_collisions()

            # --- Calculate Rewards ---
            reward += ore_mined * self.REWARD_PER_ORE
            self.score += ore_mined
            if asteroid_depleted:
                reward += self.REWARD_DEPLETE_ASTEROID
            if collision_detected:
                reward += self.PENALTY_COLLISION

            is_too_close = any(
                self.player_pos.distance_to(ast['pos']) < ast['radius'] + self.PROXIMITY_THRESHOLD
                for ast in self.asteroids
            )
            if is_too_close:
                reward += self.PENALTY_PROXIMITY

        self._update_particles()
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += self.REWARD_WIN
            if self.player_health <= 0:
                reward += self.REWARD_LOSE

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, movement):
        # Apply thrust based on movement action
        if movement == 1:  # Up
            self.player_vel.y -= self.PLAYER_ACCELERATION
            self._create_thruster_particles(math.radians(90))
        elif movement == 2:  # Down
            self.player_vel.y += self.PLAYER_ACCELERATION
            self._create_thruster_particles(math.radians(-90))
        elif movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_ACCELERATION
            self._create_thruster_particles(math.radians(0))
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_ACCELERATION
            self._create_thruster_particles(math.radians(180))

    def _update_player(self):
        # Limit speed
        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)

        # Apply friction
        self.player_vel *= self.PLAYER_FRICTION

        # Update position
        self.player_pos += self.player_vel

        # Screen wrapping
        self.player_pos.x %= self.WIDTH
        self.player_pos.y %= self.HEIGHT
        
    def _update_mining(self, is_mining):
        ore_mined_this_step = 0
        asteroid_depleted = False
        
        if not is_mining:
            return ore_mined_this_step, asteroid_depleted

        # Find closest minable asteroid
        closest_ast = None
        min_dist = self.MINING_RANGE
        for ast in self.asteroids:
            dist = self.player_pos.distance_to(ast['pos'])
            if dist < min_dist:
                min_dist = dist
                closest_ast = ast
        
        if closest_ast:
            self.mining_target = closest_ast
            # Mine ore
            mined_amount = min(closest_ast['ore'], self.MINING_RATE)
            closest_ast['ore'] -= mined_amount
            ore_mined_this_step = mined_amount
            
            # Create ore particles
            if random.random() < 0.8:
                self._create_ore_particles(closest_ast['pos'])
            
            # Check for depletion
            if closest_ast['ore'] <= 0:
                asteroid_depleted = True
                self.asteroids.remove(closest_ast)
                # Sound effect placeholder: # sfx_asteroid_depleted.play()

        return ore_mined_this_step, asteroid_depleted

    def _handle_collisions(self):
        collided = False
        for ast in self.asteroids:
            dist = self.player_pos.distance_to(ast['pos'])
            if dist < self.PLAYER_RADIUS + ast['radius']:
                self.player_health -= 1
                self._create_explosion(self.player_pos.lerp(ast['pos'], 0.5))
                # Push player away from asteroid
                if dist > 0:
                    repulsion = (self.player_pos - ast['pos']).normalize() * 5
                    self.player_vel += repulsion
                else: # Exactly on top
                    self.player_vel += pygame.Vector2(random.uniform(-5, 5), random.uniform(-5, 5))

                collided = True
                self.asteroids.remove(ast)
                # Sound effect placeholder: # sfx_explosion.play()
                break # only one collision per frame
        return collided

    def _check_termination(self):
        return (
            self.score >= self.WIN_SCORE
            or self.player_health <= 0
            or self.steps >= self.MAX_STEPS
        )

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
            "health": self.player_health,
        }

    # --- Factory Methods ---
    def _create_stars(self):
        return [(random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 2)) for _ in range(100)]

    def _create_asteroids(self):
        asteroids = []
        while len(asteroids) < self.ASTEROID_COUNT:
            radius = random.uniform(self.ASTEROID_MIN_RADIUS, self.ASTEROID_MAX_RADIUS)
            pos = pygame.Vector2(random.uniform(0, self.WIDTH), random.uniform(0, self.HEIGHT))
            
            # Avoid spawning on player
            if pos.distance_to(self.player_pos) < radius + self.PLAYER_RADIUS + 100:
                continue

            # Avoid spawning on other asteroids
            if any(pos.distance_to(ast['pos']) < radius + ast['radius'] + 10 for ast in asteroids):
                continue
                
            num_vertices = random.randint(self.ASTEROID_MIN_VERTICES, self.ASTEROID_MAX_VERTICES)
            shape = []
            for i in range(num_vertices):
                angle = (i / num_vertices) * 2 * math.pi
                dist = random.uniform(radius * 0.7, radius * 1.0)
                shape.append(pygame.Vector2(math.cos(angle) * dist, math.sin(angle) * dist))
            
            asteroids.append({
                'pos': pos,
                'radius': radius,
                'ore': radius * self.ASTEROID_ORE_DENSITY,
                'shape': shape,
                'angle': random.uniform(0, 2 * math.pi),
                'rot_speed': random.uniform(-0.01, 0.01)
            })
        return asteroids
    
    # --- Particle Methods ---
    def _create_explosion(self, pos):
        for _ in range(30):
            vel = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
            vel = vel.normalize() * random.uniform(1, 6)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': random.randint(20, 40),
                'color': random.choice(self.COLOR_EXPLOSION),
                'size': random.uniform(1, 4)
            })

    def _create_thruster_particles(self, angle_rad):
        if random.random() < 0.7:
            # Position behind the player
            offset = pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * -self.PLAYER_RADIUS
            pos = self.player_pos + offset
            
            # Velocity away from the ship
            vel_angle = angle_rad + random.uniform(-0.3, 0.3)
            vel = pygame.Vector2(math.cos(vel_angle), math.sin(vel_angle)) * -random.uniform(1, 3) + self.player_vel * 0.5

            self.particles.append({
                'pos': pos,
                'vel': vel,
                'lifespan': random.randint(10, 20),
                'color': random.choice(self.COLOR_THRUSTER),
                'size': random.uniform(1, 3)
            })

    def _create_ore_particles(self, asteroid_pos):
        direction = (self.player_pos - asteroid_pos).normalize()
        start_pos = asteroid_pos + direction * 20
        vel = direction * random.uniform(2, 4) + pygame.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        self.particles.append({
            'pos': start_pos,
            'vel': vel,
            'lifespan': random.randint(20, 35),
            'color': self.COLOR_ORE,
            'size': random.uniform(2, 4)
        })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    # --- Rendering Methods ---
    def _render_game(self):
        # Stars
        for x, y, size in self.stars:
            self.screen.set_at((x, y), (200, 200, 200) if size == 2 else (100, 100, 100))

        # Asteroids
        for ast in self.asteroids:
            ast['angle'] += ast['rot_speed']
            points = []
            for p in ast['shape']:
                rotated_p = p.rotate(math.degrees(ast['angle']))
                points.append(ast['pos'] + rotated_p)
            
            # Draw filled polygon and antialiased outline
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in points], self.COLOR_ASTEROID)
            pygame.gfxdraw.aapolygon(self.screen, [(int(p.x), int(p.y)) for p in points], self.COLOR_ASTEROID_OUTLINE)

        # Mining Beam
        if self.mining_target:
            start_pos = (int(self.player_pos.x), int(self.player_pos.y))
            end_pos = (int(self.mining_target['pos'].x), int(self.mining_target['pos'].y))
            width = int(2 + math.sin(self.steps * 0.5) * 1.5)
            pygame.draw.line(self.screen, self.COLOR_MINING_BEAM, start_pos, end_pos, width)
            pygame.gfxdraw.filled_circle(self.screen, start_pos[0], start_pos[1], width, self.COLOR_MINING_BEAM)
            pygame.gfxdraw.filled_circle(self.screen, end_pos[0], end_pos[1], width, self.COLOR_MINING_BEAM)

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), int(p['size']))

        # Player
        player_angle_deg = self.player_vel.angle_to(pygame.Vector2(1, 0)) if self.player_vel.length() > 0.1 else -90
        
        # Glow
        p1_glow = self.player_pos + pygame.Vector2(0, -self.PLAYER_RADIUS-4).rotate(-player_angle_deg)
        p2_glow = self.player_pos + pygame.Vector2(-self.PLAYER_RADIUS-2, self.PLAYER_RADIUS+2).rotate(-player_angle_deg)
        p3_glow = self.player_pos + pygame.Vector2(self.PLAYER_RADIUS+2, self.PLAYER_RADIUS+2).rotate(-player_angle_deg)
        pygame.gfxdraw.filled_trigon(self.screen, int(p1_glow.x), int(p1_glow.y), int(p2_glow.x), int(p2_glow.y), int(p3_glow.x), int(p3_glow.y), self.COLOR_PLAYER_GLOW)

        # Ship
        p1 = self.player_pos + pygame.Vector2(0, -self.PLAYER_RADIUS).rotate(-player_angle_deg)
        p2 = self.player_pos + pygame.Vector2(-self.PLAYER_RADIUS*0.8, self.PLAYER_RADIUS*0.8).rotate(-player_angle_deg)
        p3 = self.player_pos + pygame.Vector2(self.PLAYER_RADIUS*0.8, self.PLAYER_RADIUS*0.8).rotate(-player_angle_deg)
        pygame.gfxdraw.filled_trigon(self.screen, int(p1.x), int(p1.y), int(p2.x), int(p2.y), int(p3.x), int(p3.y), self.COLOR_PLAYER)
        pygame.gfxdraw.aatrigon(self.screen, int(p1.x), int(p1.y), int(p2.x), int(p2.y), int(p3.x), int(p3.y), self.COLOR_PLAYER)

    def _render_ui(self):
        # Ore count
        ore_text = self.font_ui.render(f"ORE: {int(self.score)} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(ore_text, (10, 10))

        # Health bar
        health_ratio = max(0, self.player_health / self.PLAYER_HEALTH)
        bar_width = 150
        health_bar_width = int(bar_width * health_ratio)
        
        health_color = self.COLOR_BAR_HEALTH
        if health_ratio < 0.6: health_color = self.COLOR_BAR_HEALTH_WARN
        if health_ratio < 0.3: health_color = self.COLOR_BAR_HEALTH_DANGER

        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (10, 35, bar_width, 15))
        if health_bar_width > 0:
            pygame.draw.rect(self.screen, health_color, (10, 35, health_bar_width, 15))

        # Health icons (ships)
        for i in range(self.player_health):
            icon_pos_x = 170 + i * 20
            p1 = (icon_pos_x, 38)
            p2 = (icon_pos_x - 5, 48)
            p3 = (icon_pos_x + 5, 48)
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1,p2,p3])

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                end_text = self.font_title.render("MISSION COMPLETE", True, self.COLOR_ORE)
            elif self.player_health <= 0:
                end_text = self.font_title.render("SHIP DESTROYED", True, self.COLOR_EXPLOSION[0])
            else:
                end_text = self.font_title.render("TIME UP", True, self.COLOR_TEXT)
                
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("="*30 + "\n")

    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = np.array([movement, space, shift])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Episode Finished. Final Score: {info['score']:.0f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting

        clock.tick(GameEnv.FPS)

    env.close()