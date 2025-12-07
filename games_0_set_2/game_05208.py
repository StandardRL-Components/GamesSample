import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to apply thrust, ←→ to turn. Hold space near an asteroid to mine it. Mining consumes fuel."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a space miner through an asteroid field. Collect 100 ore to win, but don't run out of fuel."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        # Colors
        self.COLOR_BG = (10, 10, 26)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_THRUSTER = (255, 165, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.ASTEROID_COLORS = {
            10: (128, 128, 128),  # Grey - Low
            20: (255, 220, 0),    # Yellow - Medium
            30: (255, 65, 54)     # Red - High
        }

        # Game parameters
        self.MAX_STEPS = 1500 # Increased for more gameplay time
        self.ORE_GOAL = 100
        self.MAX_FUEL = 500
        self.MAX_ASTEROIDS = 10
        self.PLAYER_TURN_SPEED = 5
        self.PLAYER_THRUST = 0.25
        self.PLAYER_DRAG = 0.98
        self.MINING_RANGE = 60
        self.MINING_RATE = 2 # Ore per tick
        self.FUEL_PER_MINE_TICK = 1
        self.FUEL_PER_THRUST_TICK = 0.1

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
        try:
            self.font_small = pygame.font.SysFont("Consolas", 16)
            self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 20)
            self.font_large = pygame.font.Font(None, 30)

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.fuel = None
        self.ore = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.mining_target = None
        self.mining_cooldown = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = -90 # Pointing up

        self.fuel = self.MAX_FUEL
        self.ore = 0
        self.score = 0
        self.steps = 0
        self.game_over = False

        self.asteroids = []
        self.particles = []
        self.mining_target = None
        self.mining_cooldown = 0

        for _ in range(self.MAX_ASTEROIDS):
            self._spawn_asteroid()

        if not self.stars:
            for _ in range(100):
                self.stars.append(
                    (
                        self.np_random.uniform(0, self.WIDTH),
                        self.np_random.uniform(0, self.HEIGHT),
                        self.np_random.uniform(0.5, 2.0)
                    )
                )

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        # shift_held = action[2] == 1 # Unused

        reward = 0

        # 1. Handle player input and physics
        self._handle_movement(movement)
        if movement == 1: # Thrust
            reward -= self.FUEL_PER_THRUST_TICK

        # 2. Handle mining
        reward += self._handle_mining(space_held)

        # 3. Update game state
        self._update_player_state()
        self._update_particles()

        self.steps += 1

        # 4. Check for termination
        terminated = False
        truncated = False
        if self.ore >= self.ORE_GOAL:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.fuel <= 0:
            reward -= 50
            self.fuel = 0
            terminated = True
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_movement(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 3: # Turn left
            self.player_angle -= self.PLAYER_TURN_SPEED
        if movement == 4: # Turn right
            self.player_angle += self.PLAYER_TURN_SPEED

        if movement == 1: # Thrust
            if self.fuel > 0:
                thrust_vec = pygame.math.Vector2(1, 0).rotate(self.player_angle) * self.PLAYER_THRUST
                self.player_vel += thrust_vec
                self.fuel -= self.FUEL_PER_THRUST_TICK
                # Add thruster particles
                for _ in range(2):
                    p_vel = pygame.math.Vector2(-1, 0).rotate(self.player_angle) * self.np_random.uniform(1,3)
                    p_vel.x += self.np_random.uniform(-0.5, 0.5)
                    p_vel.y += self.np_random.uniform(-0.5, 0.5)
                    self.particles.append({
                        "pos": pygame.math.Vector2(self.player_pos),
                        "vel": p_vel,
                        "lifespan": self.np_random.integers(10, 20),
                        "color": self.COLOR_THRUSTER,
                        "radius": self.np_random.uniform(2, 4)
                    })

    def _handle_mining(self, space_held):
        reward = 0
        if self.mining_cooldown > 0:
            self.mining_cooldown -= 1

        if space_held and self.fuel > 0 and self.mining_cooldown == 0:
            # Find nearest asteroid
            nearest_asteroid = None
            min_dist = float('inf')
            for asteroid in self.asteroids:
                dist = self.player_pos.distance_to(asteroid['pos'])
                if dist < min_dist:
                    min_dist = dist
                    nearest_asteroid = asteroid

            if nearest_asteroid and min_dist <= self.MINING_RANGE:
                self.mining_target = nearest_asteroid

                # Consume fuel
                self.fuel -= self.FUEL_PER_MINE_TICK
                reward -= self.FUEL_PER_MINE_TICK * 0.02

                # Add ore
                mined_amount = min(self.MINING_RATE, nearest_asteroid['ore'])
                self.ore += mined_amount
                nearest_asteroid['ore'] -= mined_amount
                reward += mined_amount * 0.1

                # Add mining particles
                for _ in range(3):
                    direction_to_player = (self.player_pos - nearest_asteroid['pos']).normalize()
                    p_pos = nearest_asteroid['pos'] + direction_to_player * nearest_asteroid['radius']
                    p_vel = direction_to_player * self.np_random.uniform(2, 4)
                    self.particles.append({
                        "pos": p_pos, "vel": p_vel, "lifespan": int(min_dist / p_vel.length()) if p_vel.length() > 0 else 20,
                        "color": nearest_asteroid['color'], "radius": self.np_random.uniform(1, 3)
                    })

                if nearest_asteroid['ore'] <= 0:
                    if nearest_asteroid['initial_ore'] == 30: # High-ore asteroid bonus
                        reward += 1
                    self.asteroids.remove(nearest_asteroid)
                    self._spawn_asteroid()
                    self.mining_target = None
            else:
                self.mining_target = None
        else:
            self.mining_target = None

        return reward

    def _update_player_state(self):
        self.player_vel *= self.PLAYER_DRAG
        self.player_pos += self.player_vel

        # Screen wrapping
        if self.player_pos.x < 0: self.player_pos.x = self.WIDTH
        if self.player_pos.x > self.WIDTH: self.player_pos.x = 0
        if self.player_pos.y < 0: self.player_pos.y = self.HEIGHT
        if self.player_pos.y > self.HEIGHT: self.player_pos.y = 0

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] *= 0.95

    def _spawn_asteroid(self):
        ore_type = self.np_random.choice([10, 20, 30], p=[0.5, 0.3, 0.2])
        radius = 10 + ore_type * 0.5

        # Ensure asteroids don't spawn on the player
        while True:
            pos = pygame.math.Vector2(
                self.np_random.uniform(0, self.WIDTH),
                self.np_random.uniform(0, self.HEIGHT)
            )
            if self.player_pos is None or pos.distance_to(self.player_pos) > 100:
                break

        self.asteroids.append({
            'pos': pos,
            'ore': ore_type,
            'initial_ore': ore_type,
            'color': self.ASTEROID_COLORS[ore_type],
            'radius': radius,
            'angle': self.np_random.uniform(0, 360),
            'vertices': self._create_asteroid_shape(radius)
        })

    def _create_asteroid_shape(self, radius):
        num_vertices = self.np_random.integers(7, 12)
        vertices = []
        for i in range(num_vertices):
            angle = (i / num_vertices) * 2 * math.pi
            dist = radius * self.np_random.uniform(0.7, 1.3)
            vertices.append(pygame.math.Vector2(math.cos(angle) * dist, math.sin(angle) * dist))
        return vertices

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, (200, 200, 220), (x, y), size)

    def _render_game(self):
        # Render asteroids
        for asteroid in self.asteroids:
            points = [asteroid['pos'] + v for v in asteroid['vertices']]
            pygame.gfxdraw.aapolygon(self.screen, [(int(p.x), int(p.y)) for p in points], asteroid['color'])
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in points], asteroid['color'])

        # Render mining beam
        if self.mining_target and self.fuel > 0:
            start_pos = self.player_pos
            end_pos = self.mining_target['pos']
            color = self.mining_target['color']
            width = int(self.np_random.uniform(1, 4))
            pygame.draw.line(self.screen, color, start_pos, end_pos, width)
            pygame.gfxdraw.aacircle(self.screen, int(end_pos.x), int(end_pos.y), int(self.mining_target['radius'])+2, (255,255,255,50))

        # Render particles
        for p in self.particles:
            if p['radius'] > 0.5:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), max(0, int(p['radius'])))

        # Render player
        self._render_player()

    def _render_player(self):
        p1 = pygame.math.Vector2(12, 0).rotate(self.player_angle) + self.player_pos
        p2 = pygame.math.Vector2(-8, 7).rotate(self.player_angle) + self.player_pos
        p3 = pygame.math.Vector2(-8, -7).rotate(self.player_angle) + self.player_pos
        points = [(int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (int(p3.x), int(p3.y))]

        # Glow effect
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), 15, (0, 100, 100, 50))


    def _render_ui(self):
        # Fuel Bar
        fuel_ratio = max(0, self.fuel / self.MAX_FUEL)
        fuel_color = (
            int(255 * (1 - fuel_ratio)),
            int(255 * fuel_ratio),
            0
        )
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, fuel_color, (10, 10, int(200 * fuel_ratio), 20))
        fuel_text = self.font_small.render("FUEL", True, self.COLOR_TEXT)
        self.screen.blit(fuel_text, (15, 12))

        # Ore Bar
        ore_ratio = min(1.0, self.ore / self.ORE_GOAL)
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 35, 200, 20))
        pygame.draw.rect(self.screen, (127, 219, 255), (10, 35, int(200 * ore_ratio), 20))
        ore_text = self.font_small.render(f"ORE: {self.ore}/{self.ORE_GOAL}", True, (10, 10, 26))
        self.screen.blit(ore_text, (15, 37))

        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0,0))
            if self.ore >= self.ORE_GOAL:
                end_text = self.font_large.render("GOAL REACHED!", True, (0, 255, 0))
            else:
                end_text = self.font_large.render("GAME OVER", True, (255, 0, 0))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fuel": self.fuel,
            "ore": self.ore,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False

    # --- Pygame setup for human play ---
    pygame.display.set_caption("Space Miner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    running = True

    print("\n" + "="*30)
    print("      SPACE MINER")
    print("="*30)
    print(env.game_description)
    print("\n" + env.user_guide)
    print("="*30 + "\n")

    while running:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1 # up
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2 # down (not used in this implementation, but available)
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3 # left
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4 # right

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                done = False

        if done:
            print(f"Episode finished. Final Info: {info}")
            # In interactive mode, reset after a delay
            pygame.time.wait(2000)
            obs, info = env.reset()


        clock.tick(env.FPS)

    env.close()
    print("Game window closed.")