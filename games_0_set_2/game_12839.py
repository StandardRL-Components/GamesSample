import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:31:41.346176
# Source Brief: brief_02839.md
# Brief Index: 2839
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player simultaneously pilots three spaceships.
    The goal is to collect energy orbs to reach a target energy level before time runs out,
    while synchronizing ship movements for bonus points.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot three spaceships at once to collect energy orbs. Synchronize their positions for a bonus and reach the target score before time runs out."
    )
    user_guide = "Use the arrow keys (↑↓←→) to pilot all three ships simultaneously."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60 # Internal simulation FPS, not necessarily render FPS

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_STAR = (180, 180, 200)
    COLOR_SHIP_RED = (255, 50, 50)
    COLOR_SHIP_GREEN = (50, 255, 50)
    COLOR_SHIP_BLUE = (50, 100, 255)
    COLOR_SHIP_OUTLINE = (255, 255, 255)
    COLOR_ORB = (255, 220, 50)
    COLOR_ORB_GLOW = (255, 220, 50, 40) # RGBA
    COLOR_SYNC_GLOW = (220, 220, 255, 20) # RGBA
    COLOR_SPARK = (255, 180, 50)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_SHADOW = (10, 10, 10)

    # Game Parameters
    SHIP_SIZE = 12
    SHIP_SPEED = 4.0
    ORB_RADIUS = 7
    INITIAL_ORBS = 20
    MAX_ORBS = 20
    SYNC_DISTANCE = 50 # Increased from brief for better gameplay
    COLLISION_PROB = 0.5
    COLLISION_RADIUS = SHIP_SIZE
    
    # Episode Parameters
    TOTAL_DURATION_SECONDS = 60
    MAX_STEPS = TOTAL_DURATION_SECONDS * 30 # Assuming 30 steps/sec for logic
    WIN_SCORE = 75

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # Initialize state variables to be defined in reset()
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        self.ships = []
        self.orbs = []
        self.stars = []
        self.particles = deque()
        self.is_synchronized = False

        self.ship_colors = [self.COLOR_SHIP_RED, self.COLOR_SHIP_GREEN, self.COLOR_SHIP_BLUE]

        # self.reset() is called by the wrapper, but we call it here for validation
        # self.reset()
        
        # Run validation check
        # self.validate_implementation() # This will fail if reset() is not called first

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.timer = self.MAX_STEPS
        self.game_over = False
        
        # Initialize Ships
        self.ships = []
        center_x, center_y = self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2
        for i in range(3):
            angle = (2 * math.pi / 3) * i
            ship_pos = pygame.Vector2(
                center_x + 60 * math.cos(angle),
                center_y + 60 * math.sin(angle)
            )
            self.ships.append({
                "pos": ship_pos,
                "energy": 0,
                "color": self.ship_colors[i]
            })

        # Initialize Orbs
        self.orbs = []
        for _ in range(self.INITIAL_ORBS):
            self._spawn_orb()

        # Initialize Starfield
        self.stars = []
        for _ in range(150):
            self.stars.append((
                self.np_random.integers(0, self.SCREEN_WIDTH),
                self.np_random.integers(0, self.SCREEN_HEIGHT),
                self.np_random.uniform(0.5, 1.5) # star size
            ))

        # Clear effects
        self.particles.clear()
        self.is_synchronized = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1
        
        # --- Game Logic ---
        reward = 0.0
        
        self._handle_movement(action)
        
        # Orb collection
        collected_this_step = self._handle_orb_collection()
        reward += 0.1 * collected_this_step
        
        # Ship collisions
        collisions_this_step = self._handle_collisions()
        reward -= 0.5 * collisions_this_step
        
        # Synchronization bonus
        if self._check_synchronization():
            reward += 0.2
        
        # Update total score
        self.score = sum(ship['energy'] for ship in self.ships)

        # --- Termination ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            terminated = True
            self.game_over = True
            reward += 100.0 # Victory reward
        elif self.timer <= 0:
            terminated = True
            self.game_over = True
            reward -= 10.0 # Timeout penalty

        self._update_particles()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, action):
        movement = action[0] # 0=none, 1=up, 2=down, 3=left, 4=right
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length_squared() > 0:
            move_vec.normalize_ip()
            move_vec *= self.SHIP_SPEED

        for ship in self.ships:
            ship["pos"] += move_vec
            # Boundary checks
            ship["pos"].x = max(self.SHIP_SIZE, min(self.SCREEN_WIDTH - self.SHIP_SIZE, ship["pos"].x))
            ship["pos"].y = max(self.SHIP_SIZE, min(self.SCREEN_HEIGHT - self.SHIP_SIZE, ship["pos"].y))

    def _handle_orb_collection(self):
        collected_count = 0
        orbs_to_remove = []
        for orb_pos in self.orbs:
            for ship in self.ships:
                if ship["pos"].distance_to(orb_pos) < self.SHIP_SIZE + self.ORB_RADIUS:
                    ship["energy"] += 1
                    collected_count += 1
                    if orb_pos not in orbs_to_remove:
                        orbs_to_remove.append(orb_pos)
                    # sfx: orb collection sound
        
        if orbs_to_remove:
            self.orbs = [orb for orb in self.orbs if orb not in orbs_to_remove]
            for _ in range(len(orbs_to_remove)):
                if len(self.orbs) < self.MAX_ORBS:
                    self._spawn_orb()
        return collected_count

    def _handle_collisions(self):
        collision_count = 0
        collided_ships = set()
        for i in range(len(self.ships)):
            for j in range(i + 1, len(self.ships)):
                ship1 = self.ships[i]
                ship2 = self.ships[j]
                if ship1["pos"].distance_to(ship2["pos"]) < self.COLLISION_RADIUS:
                    if self.np_random.random() < self.COLLISION_PROB:
                        if i not in collided_ships:
                            ship1["energy"] = max(0, ship1["energy"] - 1)
                            collided_ships.add(i)
                        if j not in collided_ships:
                            ship2["energy"] = max(0, ship2["energy"] - 1)
                            collided_ships.add(j)
                        
                        collision_count += 1
                        mid_point = (ship1["pos"] + ship2["pos"]) / 2
                        self._create_sparks(mid_point)
                        # sfx: collision sound
        return collision_count

    def _check_synchronization(self):
        max_dist = 0
        for i in range(len(self.ships)):
            for j in range(i + 1, len(self.ships)):
                dist = self.ships[i]["pos"].distance_to(self.ships[j]["pos"])
                if dist > max_dist:
                    max_dist = dist
        
        self.is_synchronized = max_dist < self.SYNC_DISTANCE
        return self.is_synchronized
    
    def _spawn_orb(self):
        pos = pygame.Vector2(
            self.np_random.integers(self.ORB_RADIUS, self.SCREEN_WIDTH - self.ORB_RADIUS),
            self.np_random.integers(self.ORB_RADIUS, self.SCREEN_HEIGHT - self.ORB_RADIUS)
        )
        self.orbs.append(pos)
        
    def _create_sparks(self, position):
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": position.copy(),
                "vel": vel,
                "lifetime": self.np_random.integers(10, 21),
                "size": self.np_random.uniform(1, 3)
            })

    def _update_particles(self):
        for _ in range(len(self.particles)):
            p = self.particles.popleft()
            p["pos"] += p["vel"]
            p["lifetime"] -= 1
            if p["lifetime"] > 0:
                self.particles.append(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Stars
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), size)
        
        # Render Synchronization Glow
        if self.is_synchronized:
            centroid = sum((s["pos"] for s in self.ships), pygame.Vector2()) / 3
            radius = self.SYNC_DISTANCE * (1.0 + 0.1 * math.sin(self.steps * 0.2))
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, self.COLOR_SYNC_GLOW, (radius, radius), radius)
            self.screen.blit(temp_surf, (centroid.x - radius, centroid.y - radius))
            
        # Render Orbs
        for pos in self.orbs:
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.ORB_RADIUS + 3, self.COLOR_ORB_GLOW)
            # Solid orb
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.ORB_RADIUS, self.COLOR_ORB)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), self.ORB_RADIUS, self.COLOR_ORB)

        # Render Ships
        for ship in self.ships:
            pos = ship["pos"]
            color = ship["color"]
            points = [
                (pos.x, pos.y - self.SHIP_SIZE),
                (pos.x - self.SHIP_SIZE / 1.5, pos.y + self.SHIP_SIZE / 2),
                (pos.x + self.SHIP_SIZE / 1.5, pos.y + self.SHIP_SIZE / 2),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_SHIP_OUTLINE)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            
        # Render Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifetime"] / 20))))
            # This needs a surface with SRCALPHA for per-pixel alpha to work
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            color = self.COLOR_SPARK
            pygame.draw.circle(temp_surf, color + (alpha,), (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (p["pos"].x - p["size"], p["pos"].y - p["size"]))


    def _render_ui(self):
        # Render Score
        score_text = f"ENERGY: {self.score}/{self.WIN_SCORE}"
        self._draw_text(score_text, (20, 15), self.font_large)

        # Render Timer
        time_left = max(0, self.timer / 30.0)
        timer_text = f"TIME: {time_left:.1f}"
        self._draw_text(timer_text, (self.SCREEN_WIDTH - 20, 15), self.font_large, align="right")
        
        # Render individual ship energy
        for i, ship in enumerate(self.ships):
            energy_text = f"{ship['energy']}"
            text_surface = self.font_small.render(energy_text, True, self.COLOR_UI_TEXT)
            text_rect = text_surface.get_rect(center=ship["pos"] + pygame.Vector2(0, self.SHIP_SIZE + 10))
            self.screen.blit(text_surface, text_rect)

    def _draw_text(self, text, position, font, align="left"):
        shadow_surface = font.render(text, True, self.COLOR_UI_SHADOW)
        text_surface = font.render(text, True, self.COLOR_UI_TEXT)
        
        shadow_rect = shadow_surface.get_rect()
        text_rect = text_surface.get_rect()

        if align == "left":
            shadow_rect.topleft = (position[0] + 2, position[1] + 2)
            text_rect.topleft = position
        elif align == "right":
            shadow_rect.topright = (position[0] + 2, position[1] + 2)
            text_rect.topright = position
        elif align == "center":
            shadow_rect.center = (position[0] + 2, position[1] + 2)
            text_rect.center = position

        self.screen.blit(shadow_surface, shadow_rect)
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "is_synchronized": self.is_synchronized,
            "ship_energies": [s['energy'] for s in self.ships]
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs, _ = self.reset()
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to test the environment
if __name__ == '__main__':
    env = GameEnv()
    env.validate_implementation() # Run validation after __init__ and first reset
    
    # --- Manual Play ---
    # Controls: Arrow keys to move. Q to quit. R to reset.
    
    obs, info = env.reset()
    done = False
    
    # Setup Pygame window for human play
    # This part is NOT headless
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tri-Sync Pilot")
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0]) # [movement, space, shift]

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()

        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action[0] = movement
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}. Resetting...")
            obs, info = env.reset()

        # Render the observation to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Limit human play to 60 FPS

    env.close()