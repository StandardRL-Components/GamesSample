import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Race enchanted flying books through a magical library.
    
    The player controls a flying book and must navigate to a delivery point
    while avoiding patrolling librarians. The book can use portals to teleport
    and create temporary clones to act as decoys.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Race enchanted flying books through a magical library. Avoid patrolling librarians, use portals, and create decoys to reach the delivery point."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to use a portal and shift to create a clone decoy."
    )
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000
    
    # Colors
    COLOR_BG = (20, 15, 40)
    COLOR_BOOKSHELF = (40, 30, 70)
    
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_GLOW = (50, 150, 255)
    
    COLOR_LIBRARIAN = (180, 40, 40)
    COLOR_LIBRARIAN_GLOW = (220, 50, 50)
    
    COLOR_PORTAL = (255, 200, 0)
    COLOR_CLONE = (100, 200, 255)
    
    COLOR_DELIVERY = (0, 200, 100)
    COLOR_DELIVERY_GLOW = (0, 255, 120)
    
    COLOR_TEXT = (230, 230, 230)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # --- Persistent State (survives reset) ---
        self.successful_deliveries = 0
        
        # --- Game State Variables (reset each episode) ---
        self.player_pos = None
        self.player_speed = 0
        self.librarians = []
        self.portals = []
        self.clones = []
        self.delivery_point = None
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.last_dist_to_target = 0
        
        self.clone_cooldown = 0
        self.portal_cooldown = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Episode State ---
        self.steps = 0
        self.score = 0
        self.clone_cooldown = 0
        self.portal_cooldown = 0
        self.particles = []
        self.clones = []

        # --- Player Setup ---
        # Unlock faster book after 3 successful deliveries
        base_speed = 5
        speed_bonus = self.successful_deliveries // 3
        self.player_speed = min(base_speed + speed_bonus, 8) # Cap speed
        self.player_pos = pygame.math.Vector2(50, self.SCREEN_HEIGHT / 2)
        
        # --- Delivery Point Setup ---
        self.delivery_point = {
            "pos": pygame.math.Vector2(self.SCREEN_WIDTH - 60, self.SCREEN_HEIGHT / 2),
            "radius": 20
        }
        self.last_dist_to_target = self.player_pos.distance_to(self.delivery_point["pos"])

        # --- Librarian Setup ---
        base_librarian_speed = 1.0
        self.librarians = [
            {
                "pos": pygame.math.Vector2(self.SCREEN_WIDTH * 0.3, self.SCREEN_HEIGHT * 0.2),
                "path_center": pygame.math.Vector2(self.SCREEN_WIDTH * 0.3, self.SCREEN_HEIGHT * 0.5),
                "path_radius": self.SCREEN_HEIGHT * 0.3,
                "angle": 0,
                "speed": base_librarian_speed,
                "radius": 12,
                "type": "circular"
            },
            {
                "pos": pygame.math.Vector2(self.SCREEN_WIDTH * 0.65, self.SCREEN_HEIGHT * 0.8),
                "start_pos": pygame.math.Vector2(self.SCREEN_WIDTH * 0.65, 100),
                "end_pos": pygame.math.Vector2(self.SCREEN_WIDTH * 0.65, self.SCREEN_HEIGHT - 100),
                "direction": 1,
                "speed": base_librarian_speed,
                "radius": 12,
                "type": "linear"
            }
        ]

        # --- Portal Setup ---
        self.portals = [
            {"pos": pygame.math.Vector2(self.SCREEN_WIDTH * 0.2, 80), "radius": 15, "link": 1},
            {"pos": pygame.math.Vector2(self.SCREEN_WIDTH * 0.8, self.SCREEN_HEIGHT - 80), "radius": 15, "link": 0}
        ]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False

        # --- Update Cooldowns ---
        if self.clone_cooldown > 0: self.clone_cooldown -= 1
        if self.portal_cooldown > 0: self.portal_cooldown -= 1

        # --- Handle Player Actions ---
        # Movement
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y = -1  # Up
        elif movement == 2: move_vec.y = 1   # Down
        elif movement == 3: move_vec.x = -1  # Left
        elif movement == 4: move_vec.x = 1   # Right
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.player_speed

        # Boundary checks
        self.player_pos.x = np.clip(self.player_pos.x, 10, self.SCREEN_WIDTH - 10)
        self.player_pos.y = np.clip(self.player_pos.y, 10, self.SCREEN_HEIGHT - 10)

        # Create Clone (Shift)
        if shift_pressed and self.clone_cooldown == 0 and len(self.clones) < 3:
            # FIX: pygame.math.Vector2 does not have a .copy() method. Instantiate a new one.
            self.clones.append({"pos": pygame.math.Vector2(self.player_pos), "lifetime": 60}) # 2 seconds at 30fps
            self.clone_cooldown = 30 # 1 second cooldown
            # Check if clone was created near a librarian for reward
            for lib in self.librarians:
                if self.player_pos.distance_to(lib["pos"]) < 100:
                    reward += 1 # Reward for tactical clone placement
                    break

        # Use Portal (Space)
        if space_pressed and self.portal_cooldown == 0:
            for i, portal in enumerate(self.portals):
                if self.player_pos.distance_to(portal["pos"]) < portal["radius"] + 5:
                    linked_portal = self.portals[portal["link"]]
                    # FIX: pygame.math.Vector2 does not have a .copy() method. Instantiate a new one.
                    self.player_pos = pygame.math.Vector2(linked_portal["pos"])
                    self.portal_cooldown = 45 # 1.5 second immunity
                    reward += 5 # Reward for using portal
                    break
        
        # --- Update Game State ---
        self.steps += 1
        self._update_librarians()
        self._update_clones()
        self._update_particles()
        
        # --- Calculate Rewards ---
        # Distance-based reward
        current_dist = self.player_pos.distance_to(self.delivery_point["pos"])
        if current_dist < self.last_dist_to_target:
            reward += 0.1
        else:
            reward -= 0.5 # Heavier penalty for moving away
        self.last_dist_to_target = current_dist

        # --- Check Terminations ---
        # Collision with Librarian
        for lib in self.librarians:
            if self.player_pos.distance_to(lib["pos"]) < lib["radius"] + 10:
                reward = -100
                terminated = True
                break
        
        # Reached Delivery Point
        if not terminated and self.player_pos.distance_to(self.delivery_point["pos"]) < self.delivery_point["radius"]:
            reward = 100
            terminated = True
            self.successful_deliveries += 1
        
        # Max steps reached
        if self.steps >= self.MAX_STEPS:
            terminated = True
            
        self.score += reward

        # Adhering to original logic where truncated is always False.
        truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_librarians(self):
        # Difficulty scaling
        speed_increase = 0
        if self.steps > 0 and self.steps % 500 == 0:
            speed_increase = 0.05
            
        for lib in self.librarians:
            lib["speed"] += speed_increase
            if lib["type"] == "circular":
                lib["angle"] += 0.02 * lib["speed"]
                lib["pos"].x = lib["path_center"].x + lib["path_radius"] * math.cos(lib["angle"])
                lib["pos"].y = lib["path_center"].y + lib["path_radius"] * math.sin(lib["angle"])
            elif lib["type"] == "linear":
                if (lib["end_pos"] - lib["start_pos"]).length() > 0:
                    direction_vec = (lib["end_pos"] - lib["start_pos"]).normalize()
                    lib["pos"] += direction_vec * lib["speed"] * lib["direction"]
                    if lib["pos"].distance_to(lib["start_pos"]) > (lib["end_pos"] - lib["start_pos"]).length() or lib["pos"].distance_to(lib["end_pos"]) > (lib["end_pos"] - lib["start_pos"]).length():
                        lib["direction"] *= -1

    def _update_clones(self):
        self.clones = [c for c in self.clones if c["lifetime"] > 0]
        for clone in self.clones:
            clone["lifetime"] -= 1

    def _update_particles(self):
        # Add new particle for player trail
        if self.steps % 2 == 0:
            # FIX: pygame.math.Vector2 does not have a .copy() method. Instantiate a new one.
            self.particles.append({
                "pos": pygame.math.Vector2(self.player_pos) + pygame.math.Vector2(random.uniform(-3, 3), random.uniform(-3, 3)),
                "radius": random.randint(3, 5),
                "lifetime": 20,
                "color": self.COLOR_PLAYER
            })
        
        self.particles = [p for p in self.particles if p["lifetime"] > 0]
        for p in self.particles:
            p["lifetime"] -= 1
            p["radius"] -= 0.2

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
            "successful_deliveries": self.successful_deliveries,
        }

    def _render_game(self):
        # --- Background Bookshelves ---
        for i in range(0, self.SCREEN_WIDTH, 80):
            pygame.draw.line(self.screen, self.COLOR_BOOKSHELF, (i, 0), (i, self.SCREEN_HEIGHT), 3)
        for i in range(0, self.SCREEN_HEIGHT, 100):
            pygame.draw.line(self.screen, self.COLOR_BOOKSHELF, (0, i), (self.SCREEN_WIDTH, i), 5)

        # --- Particles ---
        for p in self.particles:
            if p["radius"] > 0:
                self._draw_glow_circle(p["pos"], p["radius"], p["color"], p["lifetime"] / 20)

        # --- Portals ---
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 # 0 to 1
        for portal in self.portals:
            radius = portal["radius"] + pulse * 5
            self._draw_glow_circle(portal["pos"], radius, self.COLOR_PORTAL, 1.0)
            pygame.gfxdraw.filled_circle(self.screen, int(portal["pos"].x), int(portal["pos"].y), int(radius * 0.5), (0,0,0,0))

        # --- Delivery Point ---
        self._draw_glow_circle(self.delivery_point["pos"], self.delivery_point["radius"] + pulse * 5, self.COLOR_DELIVERY_GLOW, 1.0)
        pygame.draw.rect(self.screen, self.COLOR_DELIVERY, (self.delivery_point["pos"].x - 20, self.delivery_point["pos"].y + 15, 40, 10))

        # --- Clones ---
        for clone in self.clones:
            alpha = max(0, clone["lifetime"] / 60) * 200 # Fade out
            self._draw_book(clone["pos"], self.COLOR_CLONE, alpha, is_clone=True)

        # --- Librarians ---
        for lib in self.librarians:
            self._draw_glow_circle(lib["pos"], lib["radius"] + 5, self.COLOR_LIBRARIAN_GLOW, 0.7)
            pygame.draw.circle(self.screen, self.COLOR_LIBRARIAN, (int(lib["pos"].x), int(lib["pos"].y)), lib["radius"])
            pygame.draw.rect(self.screen, self.COLOR_LIBRARIAN, (lib["pos"].x - 4, lib["pos"].y - lib["radius"] - 4, 8, 4)) # Hat

        # --- Player ---
        self._draw_glow_circle(self.player_pos, 25, self.COLOR_PLAYER_GLOW, 0.5)
        self._draw_book(self.player_pos, self.COLOR_PLAYER, 255)

    def _draw_book(self, pos, color, alpha, is_clone=False):
        book_surface = pygame.Surface((24, 20), pygame.SRCALPHA)
        
        # Book cover
        pygame.draw.rect(book_surface, color, (0, 0, 20, 20))
        # Page edges
        pygame.draw.line(book_surface, (255, 255, 230), (20, 2), (23, 2), 2)
        pygame.draw.line(book_surface, (255, 255, 230), (20, 6), (23, 6), 2)
        pygame.draw.line(book_surface, (255, 255, 230), (20, 10), (23, 10), 2)
        pygame.draw.line(book_surface, (255, 255, 230), (20, 14), (23, 14), 2)
        pygame.draw.line(book_surface, (255, 255, 230), (20, 18), (23, 18), 2)
        
        book_surface.set_alpha(alpha)
        
        # Bobbing motion
        bob = math.sin(self.steps * 0.2) * 2 if not is_clone else 0
        
        self.screen.blit(book_surface, (int(pos.x - 12), int(pos.y - 10 + bob)))

    def _draw_glow_circle(self, pos, radius, color, max_alpha_factor):
        x, y = int(pos.x), int(pos.y)
        for i in range(int(radius), 0, -2):
            alpha = int(255 * (1 - (i / radius))**2 * max_alpha_factor)
            if alpha > 255: alpha = 255
            if alpha < 0: alpha = 0
            pygame.gfxdraw.filled_circle(self.screen, x, y, i, (*color, alpha))

    def _render_ui(self):
        score_text = self.font.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 30))

        deliveries_text = self.font.render(f"Deliveries: {self.successful_deliveries}", True, self.COLOR_TEXT)
        self.screen.blit(deliveries_text, (self.SCREEN_WIDTH - deliveries_text.get_width() - 10, 10))

        # Distance indicator
        if self.player_pos and self.delivery_point:
            if (self.delivery_point["pos"] - self.player_pos).length() > 0:
                target_vec = (self.delivery_point["pos"] - self.player_pos).normalize()
                indicator_pos = (self.player_pos + target_vec * 30)
                p1 = indicator_pos + target_vec.rotate(90) * 5
                p2 = indicator_pos + target_vec.rotate(-90) * 5
                p3 = indicator_pos + target_vec * 10
                pygame.draw.polygon(self.screen, self.COLOR_DELIVERY, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)])

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Example ---
    # To run this, you need to unset the dummy video driver
    # comment out: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    total_reward = 0
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Magical Library Chase")
    clock = pygame.time.Clock()

    print("\n--- Controls ---")
    print("Arrows: Move")
    print("Space: Use Portal")
    print("Shift: Create Clone")
    print("R: Reset Environment")
    print("Q: Quit")
    
    while running:
        if terminated:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            terminated = False

        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_pressed = 1 if keys[pygame.K_SPACE] else 0
        shift_pressed = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_pressed, shift_pressed]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    terminated = True # Force reset on next loop
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()