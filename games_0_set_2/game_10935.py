import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:19:54.816076
# Source Brief: brief_00935.md
# Brief Index: 935
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a player pilots a rocket to a finish line.
    The player must collect fuel orbs to power their rocket and a special boost ability,
    while dodging asteroids that damage the hull and consume fuel. The goal is to
    reach the finish line before time or fuel runs out. The environment is designed
    with a focus on high-quality retro-arcade visuals and responsive "game feel".
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a rocket to the finish line, collecting fuel orbs while dodging asteroids. "
        "Use a powerful boost to speed ahead, but watch your fuel gauge!"
    )
    user_guide = (
        "Controls: ←→ to move left and right. Press space to use a speed boost."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_TIME_SECONDS = 60
    MAX_STEPS = MAX_TIME_SECONDS * FPS

    # Colors
    COLOR_BG = (15, 20, 45)
    COLOR_ROCKET = (255, 200, 0)
    COLOR_ROCKET_GLOW = (255, 100, 0)
    COLOR_FUEL_ORB = (0, 255, 150)
    COLOR_ASTEROID = (255, 80, 80)
    COLOR_BOOST_PARTICLE = (80, 150, 255)
    COLOR_SHIELD = (50, 200, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_FINISH_LINE = (255, 255, 255)
    COLOR_UI_BAR = (40, 50, 80)
    COLOR_UI_FUEL = (0, 255, 150)

    # Game Parameters
    FINISH_LINE_X = SCREEN_WIDTH - 20
    ROCKET_SPEED = 4.0
    ROCKET_BOOST_SPEED_MULTIPLIER = 2.5
    INITIAL_FUEL = 50.0
    MAX_FUEL = 100.0
    FUEL_PER_ORB = 10.0
    FUEL_COST_ASTEROID = 20.0
    FUEL_COST_BOOST = 10.0
    INITIAL_ASTEROID_SPAWN_PERIOD = 2.0  # in seconds
    ASTEROID_SPAWN_RATE_INCREASE = 0.01 / FPS # per step
    ORBS_FOR_SHIELD = 5
    SHIELD_DURATION = 2.0 # in seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_game_over = pygame.font.SysFont('Consolas', 48, bold=True)

        # State variables are initialized in reset()
        self.rocket_pos = None
        self.fuel = None
        self.timer = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win_condition = None
        self.asteroids = None
        self.fuel_orbs = None
        self.particles = None
        self.stars = None
        self.invincibility_timer = None
        self.fuel_orbs_collected_streak = None
        self.asteroid_spawn_timer = None
        self.asteroid_spawn_period = None
        self.prev_space_held = None

        # Call reset to initialize the state
        # self.reset() # reset() is called by the environment runner

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.timer = self.MAX_TIME_SECONDS
        self.fuel = self.INITIAL_FUEL
        self.invincibility_timer = 0.0
        self.fuel_orbs_collected_streak = 0
        self.prev_space_held = False

        # Player state
        self.rocket_pos = pygame.Vector2(50, self.SCREEN_HEIGHT / 2)

        # Game objects
        self.asteroids = []
        self.fuel_orbs = []
        self.particles = []
        
        # Spawners
        self.asteroid_spawn_period = self.INITIAL_ASTEROID_SPAWN_PERIOD
        self.asteroid_spawn_timer = self.asteroid_spawn_period

        # Populate initial objects
        self._populate_stars()
        for _ in range(5):
            self._spawn_fuel_orb(random_x=True)
        for _ in range(3):
            self._spawn_asteroid(random_x=True)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        dt = 1.0 / self.FPS
        self.timer -= dt

        # --- 1. Handle Input & Update Rocket ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        is_boosting = False
        
        # Movement
        # The original code only moved the rocket with boost. This is corrected to use the movement action.
        move_direction = 0
        if movement == 3:  # Left
            move_direction = -1
        elif movement == 4:  # Right
            move_direction = 1
        
        # Boost
        if space_held and not self.prev_space_held and self.fuel >= self.FUEL_COST_BOOST:
            self.fuel -= self.FUEL_COST_BOOST
            is_boosting = True
            reward += 5.0 # Reward for using boost
            # Sound: Boost activate
            self._create_particles(self.rocket_pos, 20, self.COLOR_BOOST_PARTICLE, 3, 7, 0.5)

        self.prev_space_held = space_held
        
        rocket_current_speed = self.ROCKET_SPEED * (self.ROCKET_BOOST_SPEED_MULTIPLIER if is_boosting else 1.0)
        # Apply horizontal movement from action
        self.rocket_pos.x += move_direction * self.ROCKET_SPEED
        # Apply forward movement
        self.rocket_pos.x += rocket_current_speed * dt * 10 # Scale for better feel
        
        self.rocket_pos.x = max(10, min(self.rocket_pos.x, self.SCREEN_WIDTH - 10))
        self.rocket_pos.y = max(10, min(self.rocket_pos.y, self.SCREEN_HEIGHT - 10))


        # --- 2. Update Game Objects ---
        self._update_particles(dt)
        self._update_asteroids(dt)

        # --- 3. Handle Collisions ---
        reward += self._handle_collisions()

        # --- 4. Update Game State ---
        if self.invincibility_timer > 0:
            self.invincibility_timer -= dt
            reward += 1.0 * dt # +1 reward per second of invincibility
        else:
            self.invincibility_timer = 0
        
        self.fuel = max(0, self.fuel)

        # --- 5. Spawning ---
        self._update_spawners(dt)

        # --- 6. Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.rocket_pos.x >= self.FINISH_LINE_X:
            self.game_over = True
            self.win_condition = True
            terminated = True
            reward += 100.0
        elif self.fuel <= 0 or self.timer <= 0:
            self.game_over = True
            self.win_condition = False
            terminated = True
            reward -= 100.0
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True


        self.score += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_collisions(self):
        reward = 0
        # Rocket vs Fuel Orbs
        rocket_rect = pygame.Rect(self.rocket_pos.x - 10, self.rocket_pos.y - 10, 20, 20)
        for orb in self.fuel_orbs[:]:
            orb_rect = pygame.Rect(orb['pos'].x - 8, orb['pos'].y - 8, 16, 16)
            if rocket_rect.colliderect(orb_rect):
                self.fuel_orbs.remove(orb)
                self.fuel = min(self.MAX_FUEL, self.fuel + self.FUEL_PER_ORB)
                self.fuel_orbs_collected_streak += 1
                reward += 10.0 # Increased reward for collecting fuel
                # Sound: Fuel collected
                self._create_particles(orb['pos'], 15, self.COLOR_FUEL_ORB, 1, 4, 0.4)
                self._spawn_fuel_orb()
                if self.fuel_orbs_collected_streak >= self.ORBS_FOR_SHIELD:
                    self.invincibility_timer = self.SHIELD_DURATION
                    self.fuel_orbs_collected_streak = 0
                    # Sound: Shield activated

        # Rocket vs Asteroids
        if self.invincibility_timer <= 0:
            for asteroid in self.asteroids:
                if self.rocket_pos.distance_to(asteroid['pos']) < 10 + asteroid['size']:
                    self.fuel -= self.FUEL_COST_ASTEROID
                    reward -= 20.0 # Increased penalty for collision
                    self.fuel_orbs_collected_streak = 0 # Reset shield streak
                    self.invincibility_timer = 0.5 # Brief invulnerability after hit
                    # Sound: Asteroid collision
                    self._create_particles(self.rocket_pos, 30, self.COLOR_ASTEROID, 2, 5, 0.6)
                    break # Only one collision per frame
        return reward

    def _update_spawners(self, dt):
        # Asteroid Spawner
        self.asteroid_spawn_timer -= dt
        if self.asteroid_spawn_timer <= 0:
            self._spawn_asteroid()
            self.asteroid_spawn_period = max(0.5, self.asteroid_spawn_period - self.ASTEROID_SPAWN_RATE_INCREASE)
            self.asteroid_spawn_timer = self.asteroid_spawn_period
    
    def _spawn_asteroid(self, random_x=False):
        if len(self.asteroids) > 15: return
        x = self.SCREEN_WIDTH + 50 if not random_x else random.uniform(self.SCREEN_WIDTH * 0.3, self.SCREEN_WIDTH + 50)
        y = random.uniform(0, self.SCREEN_HEIGHT)
        pos = pygame.Vector2(x, y)
        vel = pygame.Vector2(random.uniform(-3, -1), random.uniform(-1.5, 1.5))
        size = random.uniform(15, 35)
        self.asteroids.append({'pos': pos, 'vel': vel, 'size': size, 'angle': 0, 'rot_speed': random.uniform(-2, 2)})

    def _spawn_fuel_orb(self, random_x=False):
        if len(self.fuel_orbs) > 10: return
        x = self.SCREEN_WIDTH + 30 if not random_x else random.uniform(self.SCREEN_WIDTH * 0.2, self.SCREEN_WIDTH)
        y = random.uniform(20, self.SCREEN_HEIGHT - 20)
        self.fuel_orbs.append({'pos': pygame.Vector2(x, y)})

    def _update_asteroids(self, dt):
        for asteroid in self.asteroids[:]:
            asteroid['pos'] += asteroid['vel'] * (dt * 60) # Scale velocity by dt
            asteroid['angle'] += asteroid['rot_speed'] * dt
            if asteroid['pos'].x < -50:
                self.asteroids.remove(asteroid)

    def _populate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append((
                random.randint(0, self.SCREEN_WIDTH),
                random.randint(0, self.SCREEN_HEIGHT),
                random.uniform(0.5, 1.5)
            ))

    def _create_particles(self, pos, count, color, min_speed, max_speed, lifespan):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(min_speed, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': lifespan,
                'initial_lifespan': lifespan,
                'color': color,
                'size': random.uniform(1, 3)
            })

    def _update_particles(self, dt):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= dt
            if p['lifespan'] <= 0:
                self.particles.remove(p)

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
            "fuel": self.fuel,
            "timer": self.timer,
            "invincible": self.invincibility_timer > 0,
        }

    def _render_game(self):
        # Stars
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_TEXT, (x, y), size)

        # Finish Line
        pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (self.FINISH_LINE_X, 0), (self.FINISH_LINE_X, self.SCREEN_HEIGHT), 3)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['initial_lifespan']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))

        # Fuel Orbs
        for orb in self.fuel_orbs:
            pos = (int(orb['pos'].x), int(orb['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_FUEL_ORB)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_FUEL_ORB)

        # Asteroids
        for asteroid in self.asteroids:
            self._draw_asteroid(asteroid)

        # Rocket
        if self.rocket_pos:
            self._render_rocket()

        # Game Over Text
        if self.game_over:
            text = "GOAL!" if self.win_condition else "GAME OVER"
            color = self.COLOR_FUEL_ORB if self.win_condition else self.COLOR_ASTEROID
            rendered_text = self.font_game_over.render(text, True, color)
            text_rect = rendered_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(rendered_text, text_rect)

    def _render_rocket(self):
        pos = self.rocket_pos
        # Shield effect
        if self.invincibility_timer > 0:
            pulse = abs(math.sin(self.steps * 0.3))
            radius = int(15 + pulse * 5)
            alpha = int(50 + pulse * 50)
            self._draw_glowing_circle(self.screen, self.COLOR_SHIELD, (int(pos.x), int(pos.y)), radius, 3, alpha)
        
        # Rocket body
        points = [
            pygame.Vector2(15, 0),
            pygame.Vector2(-10, -10),
            pygame.Vector2(-10, 10)
        ]
        rotated_points = [p.rotate(0) + pos for p in points]
        int_points = [(int(p.x), int(p.y)) for p in rotated_points]
        pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_ROCKET)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_ROCKET)

        # Rocket flame
        flame_pulse = 1.0 + math.sin(self.steps * 0.5) * 0.2
        flame_points = [
            pygame.Vector2(-10, -6),
            pygame.Vector2(-10, 6),
            pygame.Vector2(-15 * flame_pulse, 0)
        ]
        rotated_flame_points = [p.rotate(0) + pos for p in flame_points]
        int_flame_points = [(int(p.x), int(p.y)) for p in rotated_flame_points]
        pygame.gfxdraw.aapolygon(self.screen, int_flame_points, self.COLOR_ROCKET_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, int_flame_points, self.COLOR_ROCKET_GLOW)


    def _draw_asteroid(self, asteroid):
        points = []
        num_points = 8
        for i in range(num_points):
            angle = asteroid['angle'] + (i / num_points) * 2 * math.pi
            radius = asteroid['size'] + random.uniform(-asteroid['size']*0.2, asteroid['size']*0.2)
            p = pygame.Vector2(math.cos(angle), math.sin(angle)) * radius + asteroid['pos']
            points.append((int(p.x), int(p.y)))
        
        if len(points) > 2:
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)

    def _draw_glowing_circle(self, surface, color, center, radius, width, alpha):
        for i in range(width):
            current_radius = radius + i
            current_alpha = int(alpha * (1 - i / width))
            if current_alpha > 0 and current_radius > 0:
                pygame.gfxdraw.aacircle(surface, center[0], center[1], current_radius, (*color, current_alpha))

    def _render_ui(self):
        # Fuel UI
        fuel_text = self.font_ui.render(f"FUEL", True, self.COLOR_TEXT)
        self.screen.blit(fuel_text, (20, 10))
        fuel_bar_rect = pygame.Rect(80, 15, 150, 15)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, fuel_bar_rect)
        fuel_width = int(150 * (self.fuel / self.MAX_FUEL))
        pygame.draw.rect(self.screen, self.COLOR_UI_FUEL, (80, 15, fuel_width, 15))

        # Timer UI
        timer_text = self.font_ui.render(f"TIME: {max(0, int(self.timer))}", True, self.COLOR_TEXT)
        text_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(timer_text, text_rect)
        
        # Score UI
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 40))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Manual play example
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    # Unset the dummy video driver to allow display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    pygame.display.init()
    pygame.font.init()

    pygame.display.set_caption("Rocket Dodger")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset")
    print("Q: Quit")
    
    while True:
        # Action defaults
        movement = 0 # none
        space = 0 # released
        shift = 0 # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
                    truncated = False

        if not terminated and not truncated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space = 1
            
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation from the environment
        # The observation is (H, W, C), but pygame needs (W, H) for surface
        # and surfarray.blit_array expects (W, H, C)
        # So we transpose back
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    env.close()