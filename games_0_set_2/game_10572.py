import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:35:40.154497
# Source Brief: brief_00572.md
# Brief Index: 572
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player pilots a ship through a gravity field,
    shooting projectiles through falling geometric tubes to score points.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Pilot a ship through a gravity field, shooting projectiles through falling geometric tubes to score points."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move your ship. Press space to fire projectiles."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (10, 0, 20)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_PROJECTILE = (100, 255, 255)
    COLOR_PROJECTILE_GLOW = (50, 200, 200)
    COLOR_TUBE_SLOW = (0, 255, 100)
    COLOR_TUBE_MID = (255, 255, 0)
    COLOR_TUBE_FAST = (255, 50, 50)
    COLOR_UI = (255, 255, 255)
    
    # Player
    PLAYER_SPEED = 5.0
    PLAYER_SIZE = 15
    
    # Projectiles
    PROJECTILE_SPEED = 10.0
    PROJECTILE_RADIUS = 5
    PROJECTILE_COOLDOWN = 5  # frames
    GRAVITY = 0.4
    
    # Tubes
    TUBE_WIDTH_RANGE = (80, 150)
    TUBE_GAP_RANGE = (80, 120)
    TUBE_SPAWN_INTERVAL = 45 # frames
    TUBE_BASE_SPEED = 2.0
    TUBE_SPEED_INCREASE_INTERVAL = 500 # steps
    TUBE_SPEED_INCREASE_AMOUNT = 0.05
    TUBE_MAX_SPEED = 6.0

    # Particles
    PARTICLE_COUNT_HIT = 20
    PARTICLE_COUNT_DESTROY = 50
    PARTICLE_LIFESPAN = 25
    PARTICLE_SPEED_RANGE = (1, 4)

    # RL
    MAX_EPISODE_STEPS = 5000
    REWARD_SURVIVAL = 0.01
    REWARD_TUBE_HIT = 1.0
    REWARD_BONUS_100_SCORE = 10.0
    PENALTY_DEATH = -100.0
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State variables (initialized in reset)
        self.player_pos = None
        self.projectiles = None
        self.tubes = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.tube_speed = None
        self.tube_spawn_timer = None
        self.projectile_cooldown_timer = None
        self.prev_space_held = None
        self.bonus_100_awarded = None

        # The reset call is needed to initialize np_random, but we don't need the output yet.
        # self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50)
        self.projectiles = []
        self.tubes = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.tube_speed = self.TUBE_BASE_SPEED
        self.tube_spawn_timer = self.TUBE_SPAWN_INTERVAL
        self.projectile_cooldown_timer = 0
        self.prev_space_held = False
        self.bonus_100_awarded = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = self.REWARD_SURVIVAL

        # Update game logic
        self._update_player(movement)
        self._handle_projectile_firing(space_held)
        self._update_projectiles()
        self._update_tubes()
        self._update_particles()
        
        # Handle collisions and scoring
        reward += self._handle_collisions()

        # Check for score bonus
        if self.score >= 100 and not self.bonus_100_awarded:
            reward += self.REWARD_BONUS_100_SCORE
            self.bonus_100_awarded = True

        # Check termination conditions
        terminated = self.game_over
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if self.game_over:
            reward = self.PENALTY_DEATH
        
        self.prev_space_held = space_held
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Update Logic ---

    def _update_player(self, movement):
        if movement == 1: # Up
            self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos.y += self.PLAYER_SPEED
        elif movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.SCREEN_WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.SCREEN_HEIGHT)

    def _handle_projectile_firing(self, space_held):
        if self.projectile_cooldown_timer > 0:
            self.projectile_cooldown_timer -= 1
        
        fire_pressed = space_held and not self.prev_space_held
        if fire_pressed and self.projectile_cooldown_timer == 0:
            # SFX: Pew
            self.projectiles.append({
                "pos": self.player_pos.copy(),
                "vel": pygame.Vector2(0, -self.PROJECTILE_SPEED)
            })
            self.projectile_cooldown_timer = self.PROJECTILE_COOLDOWN

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["vel"].y += self.GRAVITY
            p["pos"] += p["vel"]
            if not self.screen.get_rect().collidepoint(p["pos"]):
                self.projectiles.remove(p)

    def _update_tubes(self):
        # Increase difficulty over time
        if self.steps % self.TUBE_SPEED_INCREASE_INTERVAL == 0 and self.steps > 0:
            self.tube_speed = min(self.TUBE_MAX_SPEED, self.tube_speed + self.TUBE_SPEED_INCREASE_AMOUNT)

        # Spawn new tubes
        self.tube_spawn_timer -= 1
        if self.tube_spawn_timer <= 0:
            self._spawn_tube()
            self.tube_spawn_timer = self.TUBE_SPAWN_INTERVAL
        
        # Move existing tubes
        for tube in self.tubes[:]:
            tube["y"] += self.tube_speed
            if tube["y"] - tube["width"] / 2 > self.SCREEN_HEIGHT:
                self.tubes.remove(tube)

    def _spawn_tube(self):
        width = self.np_random.integers(self.TUBE_WIDTH_RANGE[0], self.TUBE_WIDTH_RANGE[1])
        gap = self.np_random.integers(self.TUBE_GAP_RANGE[0], self.TUBE_GAP_RANGE[1])
        x_pos = self.np_random.integers(width // 2, self.SCREEN_WIDTH - width // 2)
        
        self.tubes.append({
            "x": x_pos,
            "y": -width / 2,
            "width": width,
            "gap": gap
        })

    def _handle_collisions(self):
        reward = 0
        
        # Projectile-Tube collisions
        for p in self.projectiles[:]:
            for tube in self.tubes[:]:
                tube_top_rect = pygame.Rect(tube["x"] - tube["width"] // 2, tube["y"] - tube["gap"] // 2 - tube["width"], tube["width"], tube["width"])
                tube_bottom_rect = pygame.Rect(tube["x"] - tube["width"] // 2, tube["y"] + tube["gap"] // 2, tube["width"], tube["width"])

                if tube_top_rect.collidepoint(p["pos"]) or tube_bottom_rect.collidepoint(p["pos"]):
                    # SFX: Hit
                    self._create_particles(p["pos"], self.PARTICLE_COUNT_HIT, self.COLOR_TUBE_SLOW)
                    self.projectiles.remove(p)
                    self.tubes.remove(tube)
                    self.score += 1
                    reward += self.REWARD_TUBE_HIT
                    break 

        # Player-Tube collisions
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE/2, self.player_pos.y - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for tube in self.tubes:
            tube_top_rect = pygame.Rect(tube["x"] - tube["width"] // 2, tube["y"] - tube["gap"] // 2 - tube["width"], tube["width"], tube["width"])
            tube_bottom_rect = pygame.Rect(tube["x"] - tube["width"] // 2, tube["y"] + tube["gap"] // 2, tube["width"], tube["width"])
            
            if player_rect.colliderect(tube_top_rect) or player_rect.colliderect(tube_bottom_rect):
                # SFX: Explosion
                self.game_over = True
                self._create_particles(self.player_pos, self.PARTICLE_COUNT_DESTROY, self.COLOR_PLAYER)
                break
        
        return reward

    # --- Particle System ---

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(self.PARTICLE_SPEED_RANGE[0], self.PARTICLE_SPEED_RANGE[1])
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": self.PARTICLE_LIFESPAN,
                "color": color
            })
    
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # friction
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    # --- Rendering ---

    def _get_observation(self):
        self._render_background()
        self._render_particles()
        self._render_tubes()
        if not self.game_over:
            self._render_player()
        self._render_projectiles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Gradient background
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_player(self):
        p = self.player_pos
        s = self.PLAYER_SIZE
        points = [
            (p.x, p.y - s),
            (p.x - s / 1.5, p.y + s / 2),
            (p.x + s / 1.5, p.y + s / 2)
        ]
        
        # Glow effect
        glow_points = [
            (p.x, p.y - s * 1.5),
            (p.x - s, p.y + s),
            (p.x + s, p.y + s)
        ]
        pygame.gfxdraw.aapolygon(self.screen, [(int(x), int(y)) for x, y in glow_points], self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(x), int(y)) for x, y in glow_points], self.COLOR_PLAYER_GLOW)
        
        # Main ship
        pygame.gfxdraw.aapolygon(self.screen, [(int(x), int(y)) for x, y in points], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(x), int(y)) for x, y in points], self.COLOR_PLAYER)

    def _render_projectiles(self):
        for p in self.projectiles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PROJECTILE_RADIUS + 2, self.COLOR_PROJECTILE_GLOW)
            # Core
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)

    def _render_tubes(self):
        # Determine color based on speed
        speed_ratio = (self.tube_speed - self.TUBE_BASE_SPEED) / (self.TUBE_MAX_SPEED - self.TUBE_BASE_SPEED)
        speed_ratio = np.clip(speed_ratio, 0, 1)
        if speed_ratio < 0.5:
            interp = speed_ratio * 2
            color = [int(c1 * (1-interp) + c2 * interp) for c1, c2 in zip(self.COLOR_TUBE_SLOW, self.COLOR_TUBE_MID)]
        else:
            interp = (speed_ratio - 0.5) * 2
            color = [int(c1 * (1-interp) + c2 * interp) for c1, c2 in zip(self.COLOR_TUBE_MID, self.COLOR_TUBE_FAST)]

        for tube in self.tubes:
            self._draw_rounded_rect(self.screen,
                pygame.Rect(tube["x"] - tube["width"] // 2, tube["y"] - tube["gap"] // 2 - tube["width"], tube["width"], tube["width"]),
                color, 10)
            self._draw_rounded_rect(self.screen,
                pygame.Rect(tube["x"] - tube["width"] // 2, tube["y"] + tube["gap"] // 2, tube["width"], tube["width"]),
                color, 10)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / self.PARTICLE_LIFESPAN))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, temp_surf.get_rect())
            self.screen.blit(temp_surf, (int(p["pos"].x - 2), int(p["pos"].y - 2)))
            
    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
    
    def _draw_rounded_rect(self, surface, rect, color, corner_radius):
        """ Draw a rectangle with rounded corners. """
        if rect.width < 2 * corner_radius or rect.height < 2 * corner_radius:
            if rect.width > 0 and rect.height > 0: # Draw plain rect if it's too small for rounding
                pygame.draw.rect(surface, color, rect)
            return

        # Top-left, top-right, bottom-left, bottom-right
        pygame.draw.circle(surface, color, (rect.left + corner_radius, rect.top + corner_radius), corner_radius)
        pygame.draw.circle(surface, color, (rect.right - corner_radius - 1, rect.top + corner_radius), corner_radius)
        pygame.draw.circle(surface, color, (rect.left + corner_radius, rect.bottom - corner_radius - 1), corner_radius)
        pygame.draw.circle(surface, color, (rect.right - corner_radius - 1, rect.bottom - corner_radius - 1), corner_radius)

        pygame.draw.rect(surface, color, (rect.left + corner_radius, rect.top, rect.width - 2 * corner_radius, rect.height))
        pygame.draw.rect(surface, color, (rect.left, rect.top + corner_radius, rect.width, rect.height - 2 * corner_radius))


    # --- Helpers ---
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tube_speed": self.tube_speed
        }
        
    def close(self):
        pygame.quit()
        
if __name__ == "__main__":
    # --- Manual Play ---
    # This part of the script is for human interaction and will not be run by the evaluator.
    # It requires a display. If you are running in a headless environment, you might need to
    # comment out this block or use a virtual display.
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display for manual play
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gravity Tube Shooter")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        # This is a manual override for human play, not part of the gym action space logic
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            total_reward = 0
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()