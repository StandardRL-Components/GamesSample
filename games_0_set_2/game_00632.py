
# Generated: 2025-08-27T14:16:51.292912
# Source Brief: brief_00632.md
# Brief Index: 632

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move selected unit. Space to cycle selection. Shift to attack."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Command a squad of survivors to fend off waves of zombies in a top-down tactical shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 5000
    NUM_WAVES = 10
    INITIAL_PLAYER_UNITS = 3
    WAVE_CLEAR_DELAY = 90  # 3 seconds at 30 FPS

    # Colors
    COLOR_BG = (20, 30, 35)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_SELECTED = (255, 255, 0)
    COLOR_ZOMBIE = (220, 20, 60)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_HEALTH_BAR_BG = (70, 0, 0)
    COLOR_HEALTH_BAR_FG = (0, 200, 0)
    COLOR_TEXT = (240, 240, 240)
    
    # Entity Properties
    PLAYER_SIZE = 12
    PLAYER_SPEED = 3
    PLAYER_HEALTH = 100
    PLAYER_ATTACK_RANGE = 150
    PLAYER_ATTACK_DAMAGE = 10
    PLAYER_ATTACK_COOLDOWN = 15 # steps

    ZOMBIE_SIZE = 8
    ZOMBIE_SPEED = 0.75
    ZOMBIE_HEALTH = 50
    ZOMBIE_ATTACK_RANGE = 12
    ZOMBIE_ATTACK_DAMAGE = 2
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 36)
        self.font_l = pygame.font.Font(None, 72)
        
        # State variables are initialized in reset()
        self.player_units = []
        self.zombies = []
        self.projectiles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.wave_number = 1
        self.wave_cleared_timer = 0
        self.selected_unit_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.wave_number = 1
        
        self.player_units = self._spawn_initial_units()
        self.zombies = []
        self.projectiles = []
        self.particles = []
        
        self.selected_unit_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        self.wave_cleared_timer = self.WAVE_CLEAR_DELAY // 2
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            shift_held = action[2] == 1

            # --- Handle Player Input ---
            self._handle_input(movement, space_held, shift_held)

            # --- Update Game Logic ---
            reward += self._update_game_state()

            self.last_space_held = space_held
            self.last_shift_held = shift_held
            
        self.steps += 1
        self.score += reward
        
        # --- Check Termination Conditions ---
        terminated = self._check_termination()
        if terminated and not self.game_won and len(self.player_units) == 0:
            reward -= 100 # Penalty for losing all units
        elif self.game_won and terminated:
            reward += 500 # Big bonus for winning the game
        
        self.score += reward # Add terminal reward to score
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_initial_units(self):
        units = []
        center_x, center_y = self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2
        for i in range(self.INITIAL_PLAYER_UNITS):
            angle = (2 * math.pi / self.INITIAL_PLAYER_UNITS) * i
            x = center_x + math.cos(angle) * 40
            y = center_y + math.sin(angle) * 40
            units.append({
                "pos": pygame.Vector2(x, y),
                "health": self.PLAYER_HEALTH,
                "cooldown": 0
            })
        return units

    def _spawn_wave(self):
        num_zombies = 8 + 2 * self.wave_number
        for _ in range(num_zombies):
            edge = self.np_random.integers(4)
            if edge == 0: # top
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -self.ZOMBIE_SIZE)
            elif edge == 1: # bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.ZOMBIE_SIZE)
            elif edge == 2: # left
                pos = pygame.Vector2(-self.ZOMBIE_SIZE, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            else: # right
                pos = pygame.Vector2(self.SCREEN_WIDTH + self.ZOMBIE_SIZE, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            
            self.zombies.append({
                "pos": pos,
                "health": self.ZOMBIE_HEALTH
            })

    def _handle_input(self, movement, space_held, shift_held):
        # Unit Selection (Space press)
        if space_held and not self.last_space_held and len(self.player_units) > 0:
            self.selected_unit_idx = (self.selected_unit_idx + 1) % len(self.player_units)

        if not self.player_units:
            return

        # Ensure selected index is valid
        if self.selected_unit_idx >= len(self.player_units):
            self.selected_unit_idx = 0
            
        selected_unit = self.player_units[self.selected_unit_idx]

        # Unit Movement
        if movement == 1: selected_unit["pos"].y -= self.PLAYER_SPEED
        elif movement == 2: selected_unit["pos"].y += self.PLAYER_SPEED
        elif movement == 3: selected_unit["pos"].x -= self.PLAYER_SPEED
        elif movement == 4: selected_unit["pos"].x += self.PLAYER_SPEED
        
        # Clamp position to screen
        selected_unit["pos"].x = max(self.PLAYER_SIZE, min(self.SCREEN_WIDTH - self.PLAYER_SIZE, selected_unit["pos"].x))
        selected_unit["pos"].y = max(self.PLAYER_SIZE, min(self.SCREEN_HEIGHT - self.PLAYER_SIZE, selected_unit["pos"].y))

        # Unit Attack (Shift press)
        if shift_held and not self.last_shift_held and selected_unit["cooldown"] == 0:
            selected_unit["cooldown"] = self.PLAYER_ATTACK_COOLDOWN
            # The actual attack logic is in _update_game_state to find the nearest target at that moment

    def _update_game_state(self):
        reward = 0

        # Update unit cooldowns
        for unit in self.player_units:
            if unit["cooldown"] > 0:
                unit["cooldown"] -= 1

        # Unit attacks
        for unit in self.player_units:
            if unit["cooldown"] == self.PLAYER_ATTACK_COOLDOWN - 1: # Attack on the frame it's triggered
                target_zombie = self._find_closest_entity(unit, self.zombies)
                if target_zombie and unit["pos"].distance_to(target_zombie["pos"]) <= self.PLAYER_ATTACK_RANGE:
                    # sfx: laser_shot
                    target_zombie["health"] -= self.PLAYER_ATTACK_DAMAGE
                    reward += 0.1
                    self.projectiles.append({"start": unit["pos"].copy(), "end": target_zombie["pos"].copy(), "life": 5})

        # Update zombies
        for zombie in self.zombies:
            target_unit = self._find_closest_entity(zombie, self.player_units)
            if target_unit:
                direction = (target_unit["pos"] - zombie["pos"]).normalize() if (target_unit["pos"] - zombie["pos"]).length() > 0 else pygame.Vector2(0,0)
                zombie["pos"] += direction * self.ZOMBIE_SPEED
                
                # Zombie attacks
                if zombie["pos"].distance_to(target_unit["pos"]) <= self.ZOMBIE_ATTACK_RANGE:
                    # sfx: zombie_bite
                    target_unit["health"] -= self.ZOMBIE_ATTACK_DAMAGE
                    self._create_particles(target_unit["pos"], self.COLOR_PLAYER, 3)


        # Update projectiles and particles
        self.projectiles = [p for p in self.projectiles if p["life"] > 0]
        for p in self.projectiles: p["life"] -= 1
        
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

        # Process deaths
        surviving_units = []
        for unit in self.player_units:
            if unit["health"] > 0:
                surviving_units.append(unit)
            else:
                # sfx: unit_death
                self._create_particles(unit["pos"], self.COLOR_PLAYER, 20, 3)
        self.player_units = surviving_units

        surviving_zombies = []
        for zombie in self.zombies:
            if zombie["health"] > 0:
                surviving_zombies.append(zombie)
            else:
                # sfx: zombie_death
                reward += 1
                self._create_particles(zombie["pos"], self.COLOR_ZOMBIE, 15, 2)
        self.zombies = surviving_zombies

        # Wave management
        if not self.zombies and not self.game_over and not self.game_won:
            if self.wave_number >= self.NUM_WAVES:
                self.game_won = True
                self.game_over = True
            elif self.wave_cleared_timer == 0:
                reward += 100
                self.wave_number += 1
                self.wave_cleared_timer = self.WAVE_CLEAR_DELAY

        if self.wave_cleared_timer > 0:
            self.wave_cleared_timer -= 1
            if self.wave_cleared_timer == 1 and not self.game_won:
                self._spawn_wave()
        
        return reward

    def _find_closest_entity(self, source, targets):
        closest_target = None
        min_dist = float('inf')
        if not targets:
            return None
        for target in targets:
            dist = source["pos"].distance_to(target["pos"])
            if dist < min_dist:
                min_dist = dist
                closest_target = target
        return closest_target

    def _create_particles(self, pos, color, count, speed_scale=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_scale
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(10, 20),
                "color": color
            })

    def _check_termination(self):
        if self.game_over:
            return True
        if not self.player_units:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            size = max(0, int(p["life"] * 0.2))
            pygame.draw.rect(self.screen, p["color"], (int(p["pos"].x - size/2), int(p["pos"].y - size/2), size, size))

        # Draw projectiles
        for p in self.projectiles:
            alpha = int(255 * (p["life"] / 5))
            color = (*self.COLOR_PROJECTILE, alpha)
            pygame.draw.aaline(self.screen, color, p["start"], p["end"], 1)

        # Draw zombies
        for i, zombie in enumerate(self.zombies):
            pos_int = (int(zombie["pos"].x), int(zombie["pos"].y))
            pygame.gfxdraw.aacircle(self.screen, *pos_int, self.ZOMBIE_SIZE, self.COLOR_ZOMBIE)
            pygame.gfxdraw.filled_circle(self.screen, *pos_int, self.ZOMBIE_SIZE, self.COLOR_ZOMBIE)
            self._render_health_bar(zombie, self.ZOMBIE_HEALTH, self.ZOMBIE_SIZE)

        # Draw player units
        for i, unit in enumerate(self.player_units):
            pos_int = (int(unit["pos"].x), int(unit["pos"].y))
            rect = pygame.Rect(pos_int[0] - self.PLAYER_SIZE/2, pos_int[1] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=2)
            
            # Selection indicator
            if i == self.selected_unit_idx:
                pygame.gfxdraw.aacircle(self.screen, *pos_int, self.PLAYER_SIZE, self.COLOR_PLAYER_SELECTED)
            
            self._render_health_bar(unit, self.PLAYER_HEALTH, self.PLAYER_SIZE)

    def _render_health_bar(self, entity, max_health, size):
        health_pct = max(0, entity["health"] / max_health)
        bar_width = size * 2
        bar_height = 4
        bar_x = entity["pos"].x - bar_width / 2
        bar_y = entity["pos"].y - size - bar_height - 5
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (bar_x, bar_y, bar_width * health_pct, bar_height))

    def _render_ui(self):
        # Top-left info
        wave_text = self.font_m.render(f"Wave: {self.wave_number}/{self.NUM_WAVES}", True, self.COLOR_TEXT)
        units_text = self.font_m.render(f"Units: {len(self.player_units)}/{self.INITIAL_PLAYER_UNITS}", True, self.COLOR_TEXT)
        score_text = self.font_m.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        self.screen.blit(units_text, (10, 40))
        self.screen.blit(score_text, (10, 70))
        
        # Center-screen messages
        if self.wave_cleared_timer > self.WAVE_CLEAR_DELAY - 30 and self.wave_number > 1:
            msg = self.font_l.render("WAVE CLEARED", True, self.COLOR_PLAYER_SELECTED)
            self.screen.blit(msg, msg.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2)))
        
        if self.game_over:
            if self.game_won:
                msg = self.font_l.render("VICTORY!", True, self.COLOR_PLAYER_SELECTED)
            else:
                msg = self.font_l.render("GAME OVER", True, self.COLOR_ZOMBIE)
            self.screen.blit(msg, msg.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "units_left": len(self.player_units),
            "zombies_left": len(self.zombies),
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

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use Pygame for human interaction
    pygame.display.set_caption("Zombie Survival RTS")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        move_action = 0 # none
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [move_action, space_action, shift_action]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset() # auto-reset
            total_reward = 0

    env.close()