import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:16:29.625549
# Source Brief: brief_00322.md
# Brief Index: 322
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a fleet of cargo ships.
    The goal is to collect cosmic resources by magnetizing them.
    The agent can clone ships to expand its fleet, but this costs resources
    and is only possible if the parent ship is not too heavy. The episode ends
    after a fixed time limit, and the final score is based on the total
    resources collected.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) for the selected ship.
    - actions[1]: Space button (0=released, 1=held) to select the next ship in the fleet.
    - actions[2]: Shift button (0=released, 1=held) to clone the selected ship.

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a fleet of cargo ships to collect cosmic resources by magnetizing them. "
        "Clone ships to expand your fleet, but watch out for clone costs and weight limits."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selected ship. "
        "Press space to cycle through your ships and press shift to clone the current ship."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 3000  # 60 seconds at 50 FPS

    # Colors
    COLOR_BG = (10, 0, 20)
    COLOR_SHIP = (255, 255, 255)
    COLOR_SHIP_GLOW = (200, 200, 255)
    COLOR_SELECTOR = (100, 255, 100)
    COLOR_BEAM = (255, 255, 0)
    RESOURCE_COLORS = [(255, 80, 80), (80, 255, 80), (80, 150, 255)]
    PARTICLE_COLORS = [(255, 255, 100), (255, 150, 50)]
    UI_BG_COLOR = (0, 0, 0, 128)
    UI_TEXT_COLOR = (255, 255, 255)
    UI_BAR_EMPTY = (50, 50, 50)
    UI_BAR_FULL = (100, 200, 255)

    # Game Parameters
    SHIP_SPEED = 4.0
    SHIP_MAX_WEIGHT = 100
    SHIP_CLONE_WEIGHT_LIMIT = 50
    CLONE_COST = 25
    MAX_RESOURCES = 25
    RESOURCE_SPAWN_INTERVAL = 45
    RESOURCE_BASE_VALUE = 5
    RESOURCE_BASE_WEIGHT = 10
    ATTRACTION_RADIUS = 120
    ATTRACTION_STRENGTH = 0.04
    COLLECTION_RADIUS = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        self.ships = []
        self.resources = []
        self.particles = []
        self.starfield = []
        self.steps = 0
        self.score = 0
        self.selected_ship_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.next_resource_spawn = 0

        self._create_starfield()
        # self.reset() # reset is called by the environment wrapper
        # self.validate_implementation() # this is for dev, not needed in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_ship_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.next_resource_spawn = 0

        # Initialize one ship
        self.ships = [{
            "pos": pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2),
            "angle": -90,
            "weight": 0
        }]

        # Initialize resources
        self.resources = []
        for _ in range(15):
            self._spawn_resource()

        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # --- Handle player input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Handle ship movement
        if len(self.ships) > 0:
            ship = self.ships[self.selected_ship_idx]
            move_vec = pygame.Vector2(0, 0)
            if movement == 1: move_vec.y = -1
            elif movement == 2: move_vec.y = 1
            elif movement == 3: move_vec.x = -1
            elif movement == 4: move_vec.x = 1

            if move_vec.length() > 0:
                ship["pos"] += move_vec.normalize() * self.SHIP_SPEED
                ship["angle"] = math.degrees(math.atan2(-move_vec.y, move_vec.x))

            # Screen wrap
            ship["pos"].x %= self.SCREEN_WIDTH
            ship["pos"].y %= self.SCREEN_HEIGHT

        # Handle ship selection (rising edge of space bar)
        if space_held and not self.last_space_held and len(self.ships) > 1:
            self.selected_ship_idx = (self.selected_ship_idx + 1) % len(self.ships)
            # sound: select_ship.wav

        # Handle cloning (rising edge of shift bar)
        if shift_held and not self.last_shift_held and len(self.ships) > 0:
            if self._attempt_clone():
                reward += 1.0
                # sound: clone_success.wav
            # else:
                # sound: clone_fail.wav

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Update game state ---
        
        # Ship-resource interactions
        for ship in self.ships:
            for res in self.resources[:]:
                dist_vec = ship["pos"] - res["pos"]
                dist = dist_vec.length()

                if dist < self.COLLECTION_RADIUS:
                    # Collect resource
                    self.score += res["value"]
                    ship["weight"] += res["weight"]
                    reward += res["value"] * 0.1
                    self.resources.remove(res)
                    self._create_particles(ship["pos"], res["color"], 15)
                    # sound: resource_collect.wav
                elif dist < self.ATTRACTION_RADIUS:
                    # Attract resource
                    res["pos"] += dist_vec.normalize() * -self.ATTRACTION_STRENGTH * (self.ATTRACTION_RADIUS - dist)
        
        # Update particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

        # Spawn new resources
        self.steps += 1
        if self.steps > self.next_resource_spawn and len(self.resources) < self.MAX_RESOURCES:
            self._spawn_resource()
            self.next_resource_spawn = self.steps + self.RESOURCE_SPAWN_INTERVAL
        
        # --- Check termination ---
        terminated = self.steps >= self.MAX_STEPS
        if terminated:
            # Final score bonus, capped as per spec
            reward += min(100.0, float(self.score))

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
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
            "ships": len(self.ships),
            "ship_weight": self.ships[self.selected_ship_idx]["weight"] if self.ships else 0
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    # --- Private Helper Methods ---

    def _attempt_clone(self):
        ship = self.ships[self.selected_ship_idx]
        if self.score >= self.CLONE_COST and ship["weight"] <= self.SHIP_CLONE_WEIGHT_LIMIT:
            self.score -= self.CLONE_COST
            
            offset = pygame.Vector2(random.uniform(-30, 30), random.uniform(-30, 30))
            new_ship_pos = ship["pos"] + offset
            
            self.ships.append({
                "pos": new_ship_pos,
                "angle": ship["angle"],
                "weight": 0
            })
            self._create_particles(ship["pos"], self.COLOR_SHIP_GLOW, 30, speed=3)
            return True
        return False

    def _spawn_resource(self):
        pos = pygame.Vector2(
            random.randint(20, self.SCREEN_WIDTH - 20),
            random.randint(20, self.SCREEN_HEIGHT - 20)
        )
        self.resources.append({
            "pos": pos,
            "color": random.choice(self.RESOURCE_COLORS),
            "value": self.RESOURCE_BASE_VALUE,
            "weight": self.RESOURCE_BASE_WEIGHT,
            "size": random.randint(5, 8)
        })

    def _create_starfield(self):
        self.starfield = []
        for _ in range(150):
            self.starfield.append((
                random.randint(0, self.SCREEN_WIDTH),
                random.randint(0, self.SCREEN_HEIGHT),
                random.randint(1, 2),
                random.randint(50, 150)
            ))

    def _create_particles(self, pos, color, count, speed=2.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(0.5, speed)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": random.randint(20, 40),
                "color": color,
                "size": random.randint(1, 3)
            })

    # --- Rendering Methods ---

    def _render_game(self):
        # Draw stars
        for x, y, size, bright in self.starfield:
            pygame.draw.circle(self.screen, (bright, bright, bright), (x, y), size)

        # Draw attraction beams and resources
        for res in self.resources:
            pygame.gfxdraw.filled_circle(self.screen, int(res["pos"].x), int(res["pos"].y), res["size"], res["color"])
            pygame.gfxdraw.aacircle(self.screen, int(res["pos"].x), int(res["pos"].y), res["size"], res["color"])
            for ship in self.ships:
                if ship["pos"].distance_to(res["pos"]) < self.ATTRACTION_RADIUS:
                    beam_color = (*self.COLOR_BEAM, 100)
                    pygame.draw.aaline(self.screen, beam_color, ship["pos"], res["pos"], 1)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 40))
            color = (*p["color"], alpha)
            try:
                # Use SRCALPHA surface for proper alpha blending
                temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
                self.screen.blit(temp_surf, (p["pos"][0] - p["size"], p["pos"][1] - p["size"]))
            except (TypeError, ValueError):
                 # Fallback for invalid colors during fade
                 pass
        
        # Draw ships and selector
        for i, ship in enumerate(self.ships):
            if i == self.selected_ship_idx:
                self._draw_ship(self.screen, ship, is_selected=True)
            else:
                self._draw_ship(self.screen, ship, is_selected=False)

    def _draw_ship(self, surface, ship, is_selected):
        pos = ship["pos"]
        angle_rad = math.radians(ship["angle"])
        
        # Points for a triangle
        p1 = pos + pygame.Vector2(12, 0).rotate(ship["angle"])
        p2 = pos + pygame.Vector2(-8, 7).rotate(ship["angle"])
        p3 = pos + pygame.Vector2(-8, -7).rotate(ship["angle"])
        points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]

        # Glow effect
        glow_radius = 20
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_SHIP_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        surface.blit(glow_surf, (int(pos.x - glow_radius), int(pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Ship body
        pygame.gfxdraw.aapolygon(surface, points, self.COLOR_SHIP)
        pygame.gfxdraw.filled_polygon(surface, points, self.COLOR_SHIP)
        
        # Selector ring
        if is_selected:
            pygame.gfxdraw.aacircle(surface, int(pos.x), int(pos.y), 22, self.COLOR_SELECTOR)
            pygame.gfxdraw.aacircle(surface, int(pos.x), int(pos.y), 23, self.COLOR_SELECTOR)

    def _render_ui(self):
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
        ui_surf.fill(self.UI_BG_COLOR)
        self.screen.blit(ui_surf, (0, 0))

        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.UI_TEXT_COLOR)
        self.screen.blit(score_text, (10, 10))

        # Time
        time_left = (self.MAX_STEPS - self.steps) / 50 # Assuming 50fps for display
        time_text = self.font_small.render(f"TIME: {time_left:.1f}s", True, self.UI_TEXT_COLOR)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - 120, 15))

        # Ship Info
        if len(self.ships) > 0:
            ship = self.ships[self.selected_ship_idx]
            ship_info_text = self.font_small.render(f"SHIP {self.selected_ship_idx + 1}/{len(self.ships)}", True, self.UI_TEXT_COLOR)
            self.screen.blit(ship_info_text, (220, 15))

            # Weight Bar
            weight_pct = min(1.0, ship["weight"] / self.SHIP_MAX_WEIGHT)
            bar_width = 150
            bar_height = 15
            bar_x = 320
            bar_y = 15
            
            pygame.draw.rect(self.screen, self.UI_BAR_EMPTY, (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.UI_BAR_FULL, (bar_x, bar_y, int(bar_width * weight_pct), bar_height))
            
            # Clone limit indicator
            clone_limit_x = bar_x + int(bar_width * (self.SHIP_CLONE_WEIGHT_LIMIT / self.SHIP_MAX_WEIGHT))
            pygame.draw.line(self.screen, (255, 100, 100), (clone_limit_x, bar_y), (clone_limit_x, bar_y + bar_height), 2)
            
            weight_text = self.font_small.render(f"{ship['weight']}/{self.SHIP_MAX_WEIGHT}", True, self.UI_TEXT_COLOR)
            self.screen.blit(weight_text, (bar_x + bar_width + 10, 15))


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" 
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Cosmic Cloner")
        clock = pygame.time.Clock()
        
        terminated = False
        total_reward = 0
        
        print("\n--- Manual Control ---")
        print("Arrows: Move Ship")
        print("Space: Select Next Ship")
        print("Shift: Clone Selected Ship")
        print("Q: Quit")
        
        while not terminated:
            movement = 0  # no-op
            space = 0
            shift = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    terminated = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(50) # Limit frame rate for playability

        print(f"\nGame Over!")
        print(f"Final Score: {info['score']}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Total Steps: {info['steps']}")
        
        env.close()
    except pygame.error as e:
        print(f"\nCould not run in graphical mode: {e}")
        print("This is expected in a headless environment. The environment code is likely correct.")