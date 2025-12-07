# Generated: 2025-08-27T23:27:57.106757
# Source Brief: brief_03480.md
# Brief Index: 3480

        
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
        "Controls: Arrow keys to move the crosshair. Press space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hit fast-moving targets in an isometric 2D arcade environment before running out of ammo."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Screen dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GRID = (35, 40, 60)
        self.COLOR_CROSSHAIR = (255, 0, 80)
        self.COLOR_TARGET = (230, 40, 40)
        self.COLOR_TARGET_SHADOW = (10, 12, 22)
        self.COLOR_PROJECTILE = (0, 255, 150)
        self.COLOR_HIT_EFFECT = (0, 150, 255)
        self.COLOR_MISS_EFFECT = (255, 200, 0)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_AMMO = (255, 200, 0)
        self.COLOR_UI_HITS = (0, 255, 150)
        self.COLOR_CANNON_FLASH = (255, 255, 255)

        # Fonts
        try:
            self.font_ui = pygame.font.Font(None, 28)
            self.font_game_over = pygame.font.Font(None, 72)
        except IOError:
            self.font_ui = pygame.font.SysFont("monospace", 20)
            self.font_game_over = pygame.font.SysFont("monospace", 50)
        
        # Game constants
        self.LOGICAL_WIDTH = 300
        self.LOGICAL_HEIGHT = 300
        self.MAX_STEPS = 1000
        self.WIN_CONDITION = 15
        self.STARTING_AMMO = 10
        self.CROSSHAIR_SPEED = 5
        self.PROJECTILE_SPEED = 15
        self.BASE_TARGET_SPEED = 1.0
        self.TARGET_SPEED_INCREASE = 0.05
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.crosshair_pos = None
        self.targets = []
        self.projectiles = []
        self.particles = []
        self.ammo = 0
        self.targets_hit = 0
        self.prev_space_held = False
        self.rng = np.random.default_rng()

        self.reset()
        # self.validate_implementation() # Optional validation call

    def _setup_constants(self):
        self.ISO_OFFSET_X = self.SCREEN_WIDTH // 2
        self.ISO_OFFSET_Y = 120
        self.ISO_SCALE = 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self._setup_constants()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.crosshair_pos = pygame.Vector2(self.LOGICAL_WIDTH / 2, self.LOGICAL_HEIGHT / 2)
        
        self.targets = []
        self.projectiles = []
        self.particles = []
        
        self.ammo = self.STARTING_AMMO
        self.targets_hit = 0
        self.prev_space_held = True # Prevent firing on first frame
        
        self._spawn_targets(5)
        
        return self._get_observation(), self._get_info()

    def _spawn_targets(self, num_targets):
        for _ in range(num_targets):
            path_type = self.rng.choice(["line_x", "line_y", "circle"])
            start_phase = self.rng.random() * 2 * math.pi
            
            if path_type == "line_x":
                y_pos = self.rng.integers(50, self.LOGICAL_HEIGHT - 50)
                path_func = lambda t: pygame.Vector2(self.LOGICAL_WIDTH / 2 + math.sin(t) * (self.LOGICAL_WIDTH/2 - 20), y_pos)
            elif path_type == "line_y":
                x_pos = self.rng.integers(50, self.LOGICAL_WIDTH - 50)
                path_func = lambda t: pygame.Vector2(x_pos, self.LOGICAL_HEIGHT / 2 + math.cos(t) * (self.LOGICAL_HEIGHT/2 - 20))
            else: # circle
                radius = self.rng.integers(40, int(min(self.LOGICAL_WIDTH, self.LOGICAL_HEIGHT) / 2.5))
                path_func = lambda t: pygame.Vector2(
                    self.LOGICAL_WIDTH / 2 + math.sin(t) * radius,
                    self.LOGICAL_HEIGHT / 2 + math.cos(t) * radius
                )

            self.targets.append({
                "path_func": path_func,
                "phase": start_phase,
                "pos": path_func(start_phase),
                "radius": 10
            })

    def step(self, action):
        reward = 0
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            # shift_held = action[2] == 1 # Unused

            self._handle_input(movement, space_held)
            reward += self._update_game_state()
            
            self.steps += 1
            terminated = self._check_termination()

            if terminated and not self.game_over:
                if self.targets_hit >= self.WIN_CONDITION:
                    reward += 10 # Win bonus
                else:
                    reward -= 10 # Lose penalty
                self.game_over = True
        else:
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Move crosshair
        if movement == 1: self.crosshair_pos.y -= self.CROSSHAIR_SPEED
        elif movement == 2: self.crosshair_pos.y += self.CROSSHAIR_SPEED
        elif movement == 3: self.crosshair_pos.x -= self.CROSSHAIR_SPEED
        elif movement == 4: self.crosshair_pos.x += self.CROSSHAIR_SPEED
        
        # Clamp crosshair to logical area
        self.crosshair_pos.x = max(0, min(self.LOGICAL_WIDTH, self.crosshair_pos.x))
        self.crosshair_pos.y = max(0, min(self.LOGICAL_HEIGHT, self.crosshair_pos.y))

        # Fire projectile on key press (not hold)
        if space_held and not self.prev_space_held and self.ammo > 0:
            self.ammo -= 1
            # SFX: Pew
            cannon_pos = pygame.Vector2(self.LOGICAL_WIDTH / 2, -20)
            direction = (self.crosshair_pos - cannon_pos).normalize()
            self.projectiles.append({
                "pos": pygame.Vector2(cannon_pos),
                "vel": direction * self.PROJECTILE_SPEED,
                "radius": 5,
            })
            self._create_particles(10, self._iso_transform(cannon_pos), self.COLOR_CANNON_FLASH, 10, 3)

        self.prev_space_held = space_held

    def _update_game_state(self):
        step_reward = 0

        # Update targets
        current_target_speed = self.BASE_TARGET_SPEED + self.targets_hit * self.TARGET_SPEED_INCREASE
        for target in self.targets:
            target["phase"] += 0.02 * current_target_speed
            target["pos"] = target["path_func"](target["phase"])

        # Update projectiles and check for collisions
        projectiles_to_remove = []
        targets_to_remove = []
        for i, proj in enumerate(self.projectiles):
            proj["pos"] += proj["vel"]
            
            # Check for target hits
            hit = False
            for j, target in enumerate(self.targets):
                if proj["pos"].distance_to(target["pos"]) < target["radius"] + proj["radius"]:
                    if i not in projectiles_to_remove and j not in targets_to_remove:
                        # SFX: Explosion
                        self._create_particles(30, self._iso_transform(target["pos"]), self.COLOR_HIT_EFFECT, 20, 5)
                        projectiles_to_remove.append(i)
                        targets_to_remove.append(j)
                        self.targets_hit += 1
                        step_reward += 1.1 # +1 for event, +0.1 for hit
                        hit = True
                        break # Projectile can only hit one target
            
            # Check for out of bounds (miss)
            if not hit and not (0 <= proj["pos"].x <= self.LOGICAL_WIDTH and 0 <= proj["pos"].y <= self.LOGICAL_HEIGHT):
                if i not in projectiles_to_remove:
                    # SFX: Fizzle
                    self._create_particles(5, self._iso_transform(proj["pos"]), self.COLOR_MISS_EFFECT, 5, 2)
                    projectiles_to_remove.append(i)
                    step_reward -= 0.01

        # Remove hit/missed entities
        self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
        self.targets = [t for i, t in enumerate(self.targets) if i not in targets_to_remove]
        
        # Respawn targets to maintain count
        if len(targets_to_remove) > 0:
            self._spawn_targets(len(targets_to_remove))

        # Update particles
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1

        return step_reward

    def _check_termination(self):
        return (
            self.targets_hit >= self.WIN_CONDITION or
            (self.ammo <= 0 and not self.projectiles) or
            self.steps >= self.MAX_STEPS
        )

    def _iso_transform(self, pos, z=0):
        iso_x = (pos.x - pos.y) * self.ISO_SCALE + self.ISO_OFFSET_X
        iso_y = (pos.x + pos.y) * 0.5 * self.ISO_SCALE + self.ISO_OFFSET_Y - z
        return int(iso_x), int(iso_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background_grid()
        
        # Sort entities by Y-position for correct isometric layering
        renderable_entities = []
        for target in self.targets:
            renderable_entities.append(("target", target))
        for proj in self.projectiles:
            renderable_entities.append(("projectile", proj))
        
        # Add crosshair as a special entity that doesn't need Z-sorting based on its own Y
        # It should be sorted based on where it logically is on the ground plane
        renderable_entities.append(("crosshair", {"pos": self.crosshair_pos}))

        renderable_entities.sort(key=lambda item: item[1]["pos"].y)
        
        for entity_type, data in renderable_entities:
            if entity_type == "target":
                iso_pos = self._iso_transform(data["pos"])
                shadow_pos = self._iso_transform(data["pos"], z=-5)
                pygame.gfxdraw.filled_ellipse(self.screen, shadow_pos[0], shadow_pos[1] + 10, 12, 6, self.COLOR_TARGET_SHADOW)
                pygame.gfxdraw.aacircle(self.screen, iso_pos[0], iso_pos[1], 10, self.COLOR_TARGET)
                pygame.gfxdraw.filled_circle(self.screen, iso_pos[0], iso_pos[1], 10, self.COLOR_TARGET)
            elif entity_type == "projectile":
                iso_pos = self._iso_transform(data["pos"], z=5)
                pygame.gfxdraw.aacircle(self.screen, iso_pos[0], iso_pos[1], 5, self.COLOR_PROJECTILE)
                pygame.gfxdraw.filled_circle(self.screen, iso_pos[0], iso_pos[1], 5, self.COLOR_PROJECTILE)
            elif entity_type == "crosshair":
                iso_pos = self._iso_transform(data["pos"])
                size = 10 + math.sin(self.steps * 0.2) * 2
                pygame.draw.line(self.screen, self.COLOR_CROSSHAIR, (iso_pos[0] - size, iso_pos[1]), (iso_pos[0] + size, iso_pos[1]), 2)
                pygame.draw.line(self.screen, self.COLOR_CROSSHAIR, (iso_pos[0], iso_pos[1] - size), (iso_pos[0], iso_pos[1] + size), 2)

        self._render_particles()

    def _render_background_grid(self):
        for i in range(0, self.LOGICAL_WIDTH + 1, 20):
            start_iso = self._iso_transform(pygame.Vector2(i, 0))
            end_iso = self._iso_transform(pygame.Vector2(i, self.LOGICAL_HEIGHT))
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_iso, end_iso)
        for i in range(0, self.LOGICAL_HEIGHT + 1, 20):
            start_iso = self._iso_transform(pygame.Vector2(0, i))
            end_iso = self._iso_transform(pygame.Vector2(self.LOGICAL_WIDTH, i))
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_iso, end_iso)

    def _create_particles(self, count, pos, color, lifespan, speed_max):
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * speed_max
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "lifespan": lifespan + self.rng.integers(0, 5),
                "color": color,
                "size": self.rng.integers(2, 5)
            })

    def _render_particles(self):
        for p in self.particles:
            size = int(p["size"] * (p["lifespan"] / 20))
            if size > 0:
                pos = (int(p["pos"].x), int(p["pos"].y))
                pygame.draw.rect(self.screen, p["color"], (*pos, size, size))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Hits
        hits_text = self.font_ui.render(f"HITS: {self.targets_hit}/{self.WIN_CONDITION}", True, self.COLOR_UI_HITS)
        self.screen.blit(hits_text, (self.SCREEN_WIDTH - hits_text.get_width() - 10, 10))
        
        # Ammo
        ammo_text = self.font_ui.render(f"AMMO: {self.ammo}", True, self.COLOR_UI_AMMO)
        self.screen.blit(ammo_text, (self.SCREEN_WIDTH - ammo_text.get_width() - 10, self.SCREEN_HEIGHT - ammo_text.get_height() - 10))

        # Game Over Message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, (0,0))
            if self.targets_hit >= self.WIN_CONDITION:
                end_text = self.font_game_over.render("YOU WIN!", True, self.COLOR_UI_HITS)
            else:
                end_text = self.font_game_over.render("GAME OVER", True, self.COLOR_TARGET)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ammo": self.ammo,
            "targets_hit": self.targets_hit,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # The main loop is for human play and requires a display.
    # Unset the dummy video driver if you want to play manually.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # For human play
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Pygame setup for human play
    pygame.display.set_caption("Isometric Shooter")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    action = env.action_space.sample()
    action[0] = 0 # Start with no movement
    action[1] = 0 # Start with space released
    action[2] = 0 # Start with shift released

    while not done:
        # Human input
        movement_action = 0
        space_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1

        action = [movement_action, space_action, 0]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control FPS
        env.clock.tick(30)
        
        if done:
            print(f"Game Over. Final Score: {info['score']}, Hits: {info['targets_hit']}")
            # Wait a bit before resetting to show the final screen
            pygame.time.wait(3000)
            obs, info = env.reset()
            done = False


    env.close()