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
    Centipede Exterminator Tycoon
    
    A resource management game where the player manages a centipede-hunting business.
    The goal is to reach a net worth of 10,000 gold by hiring hunters and upgrading
    their equipment, balancing costs with income from extermination.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = "Manage a centipede-hunting business by hiring hunters and upgrading equipment to increase your net worth."
    user_guide = "Controls: Press ↑ to hire a new hunter. Press space to upgrade all equipment."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.WIN_NET_WORTH = 10000
        
        # Costs and rates
        self.HUNTER_COST = 50
        self.BASE_UPGRADE_COST = 50
        self.BASE_HUNT_RATE = 0.5  # Base centipedes per hunter per level
        self.GOLD_PER_CENTIPEDE = 1
        self.BASE_REPRODUCTION_RATE = 0.01

        # Visuals
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GROUND = (44, 62, 80)
        self.COLOR_HUNTER = (46, 204, 113)
        self.COLOR_HUNTER_GLOW = (46, 204, 113, 50)
        self.COLOR_CENTIPEDE = (231, 76, 60)
        self.COLOR_UI_BG = (52, 73, 94, 200)
        self.COLOR_TEXT = (236, 240, 241)
        self.COLOR_GOLD = (241, 196, 15)
        self.COLOR_POSITIVE = (39, 174, 96)
        self.COLOR_NEGATIVE = (192, 57, 43)

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
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gold = 0
        self.num_hunters = 0
        self.equipment_level = 0
        self.centipede_population = 0
        self.last_action_feedback = ""
        self.floating_texts = []
        self.hunters = []
        self.centipede_particles = []
        
        # Initialize state
        # self.reset() is called by the wrapper, no need to call it here.
        
    def _get_upgrade_cost(self):
        return self.BASE_UPGRADE_COST * self.equipment_level

    def _calculate_net_worth(self):
        # Value of gold + value of hunters + value of equipment upgrades
        hunter_value = self.num_hunters * self.HUNTER_COST
        equipment_value = sum(self.BASE_UPGRADE_COST * i for i in range(1, self.equipment_level))
        return self.gold + hunter_value + equipment_value

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.game_over = False
        self.win_condition_met = False
        
        self.gold = 100
        self.num_hunters = 1
        self.equipment_level = 1
        self.centipede_population = 500
        
        self.last_action_feedback = "New business day!"
        self.floating_texts = []
        self._respawn_visuals()

        self.score = self._calculate_net_worth()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        reward = 0
        self.last_action_feedback = "Waiting..."

        # 1. Handle player actions (prioritize upgrades over hires if both are attempted)
        action_taken = False
        if space_held:
            upgrade_cost = self._get_upgrade_cost()
            if self.gold >= upgrade_cost:
                self.gold -= upgrade_cost
                self.equipment_level += 1
                reward += 0.5
                self.last_action_feedback = f"Equipment upgraded to Lvl {self.equipment_level}!"
                self._add_floating_text(f"-{upgrade_cost}G", (self.WIDTH // 2, 50), self.COLOR_NEGATIVE)
                action_taken = True
            else:
                self.last_action_feedback = "Not enough gold to upgrade!"
                action_taken = True
        
        if not action_taken and movement == 1: # Hire Hunter
            if self.gold >= self.HUNTER_COST:
                self.gold -= self.HUNTER_COST
                self.num_hunters += 1
                self._add_hunter_visual()
                reward += 1.0
                self.last_action_feedback = f"Hired a new hunter! Total: {self.num_hunters}"
                self._add_floating_text(f"-{self.HUNTER_COST}G", (self.WIDTH // 2, 50), self.COLOR_NEGATIVE)
            else:
                self.last_action_feedback = "Not enough gold to hire!"

        # 2. Update game logic for the "day"
        # Hunting phase
        hunts_per_hunter = self.BASE_HUNT_RATE * self.equipment_level
        centipedes_hunted = int(self.num_hunters * hunts_per_hunter)
        centipedes_hunted = min(self.centipede_population, centipedes_hunted)
        
        if centipedes_hunted > 0:
            self.centipede_population -= centipedes_hunted
            gold_earned = centipedes_hunted * self.GOLD_PER_CENTIPEDE
            self.gold += gold_earned
            reward += 0.1 * centipedes_hunted
            self._add_floating_text(f"+{gold_earned}G", (self.WIDTH // 2, 50), self.COLOR_GOLD)
            # Visual effect for hunting
            for _ in range(min(centipedes_hunted, 20)): # Limit particles for performance
                if self.centipede_particles:
                    self.centipede_particles.pop(self.np_random.integers(len(self.centipede_particles)))

        # Reproduction phase
        repro_rate = self.BASE_REPRODUCTION_RATE + (self.steps // 100 * 0.0001)
        new_centipedes = int(self.centipede_population * repro_rate)
        self.centipede_population += new_centipedes
        # Add visual particles for new centipedes
        for _ in range(min(new_centipedes, 20)):
            self._add_centipede_particle()

        # 3. Update internal state
        self.steps += 1
        self.score = self._calculate_net_worth()
        self._update_animations()

        # 4. Check for termination
        terminated = False
        truncated = False
        if self.score >= self.WIN_NET_WORTH:
            reward += 100
            terminated = True
            self.game_over = True
            self.win_condition_met = True
            self.last_action_feedback = "VICTORY! You are the ultimate exterminator!"
        elif self.gold < 0: # Check for bankruptcy
            reward -= 100
            terminated = True
            self.game_over = True
            self.last_action_feedback = "DEFEAT! Your business is bankrupt."
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
            self.last_action_feedback = "TIME'S UP! The business has closed."

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "hunters": self.num_hunters,
            "equipment_level": self.equipment_level,
            "centipedes": self.centipede_population
        }

    def _render_game(self):
        # Draw hunting grounds
        ground_rect = pygame.Rect(0, self.HEIGHT * 0.4, self.WIDTH, self.HEIGHT * 0.6)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)

        # Draw centipede particles
        for p in self.centipede_particles:
            p['pos'][0] += self.np_random.uniform(-0.5, 0.5)
            p['pos'][1] += self.np_random.uniform(-0.5, 0.5)
            p['pos'][0] = np.clip(p['pos'][0], 0, self.WIDTH)
            p['pos'][1] = np.clip(p['pos'][1], ground_rect.top, self.HEIGHT)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, self.COLOR_CENTIPEDE)
        
        # Draw hunters
        for h in self.hunters:
            # Glow effect
            glow_radius = 15 + self.equipment_level
            surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, self.COLOR_HUNTER_GLOW, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(surf, (int(h['pos'][0] - glow_radius), int(h['pos'][1] - glow_radius)))
            
            # Hunter body
            pygame.draw.circle(self.screen, self.COLOR_HUNTER, (int(h['pos'][0]), int(h['pos'][1])), 8)

    def _render_ui(self):
        # UI Panel
        ui_panel = pygame.Surface((self.WIDTH, 110), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))
        pygame.draw.line(self.screen, self.COLOR_TEXT, (0, 110), (self.WIDTH, 110), 2)
        
        # Stats
        gold_text = self.font_medium.render(f"GOLD: {int(self.gold)}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (15, 10))

        hunters_text = self.font_medium.render(f"HUNTERS: {self.num_hunters}", True, self.COLOR_POSITIVE)
        self.screen.blit(hunters_text, (15, 40))

        equip_text = self.font_medium.render(f"EQUIP LVL: {self.equipment_level}", True, self.COLOR_TEXT)
        self.screen.blit(equip_text, (15, 70))
        
        centipedes_text = self.font_medium.render(f"PESTS: {self.centipede_population}", True, self.COLOR_NEGATIVE)
        text_rect = centipedes_text.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(centipedes_text, text_rect)

        day_text = self.font_medium.render(f"DAY: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        text_rect = day_text.get_rect(topright=(self.WIDTH - 15, 40))
        self.screen.blit(day_text, text_rect)

        score_text = self.font_medium.render(f"WORTH: {int(self.score)}", True, self.COLOR_GOLD)
        text_rect = score_text.get_rect(topright=(self.WIDTH - 15, 70))
        self.screen.blit(score_text, text_rect)

        # Action hints
        hunter_cost_color = self.COLOR_POSITIVE if self.gold >= self.HUNTER_COST else self.COLOR_NEGATIVE
        upgrade_cost_color = self.COLOR_POSITIVE if self.gold >= self._get_upgrade_cost() else self.COLOR_NEGATIVE
        
        hire_text = self.font_small.render(f"[↑] HIRE HUNTER ({self.HUNTER_COST}G)", True, hunter_cost_color)
        hire_rect = hire_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 45))
        self.screen.blit(hire_text, hire_rect)

        upgrade_text = self.font_small.render(f"[SPACE] UPGRADE ({self._get_upgrade_cost()}G)", True, upgrade_cost_color)
        upgrade_rect = upgrade_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 25))
        self.screen.blit(upgrade_text, upgrade_rect)

        # Floating texts
        for ft in self.floating_texts:
            text_surf = self.font_small.render(ft['text'], True, ft['color'])
            text_surf.set_alpha(ft['alpha'])
            self.screen.blit(text_surf, (int(ft['pos'][0]), int(ft['pos'][1])))

        # Last action feedback
        feedback_surf = self.font_small.render(self.last_action_feedback, True, self.COLOR_TEXT)
        feedback_rect = feedback_surf.get_rect(center=(self.WIDTH // 2, 130))
        self.screen.blit(feedback_surf, feedback_rect)
        
        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "VICTORY" if self.win_condition_met else "GAME OVER"
            status_color = self.COLOR_POSITIVE if self.win_condition_met else self.COLOR_NEGATIVE
            
            status_surf = self.font_large.render(status_text, True, status_color)
            status_rect = status_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 20))
            self.screen.blit(status_surf, status_rect)
            
            reset_surf = self.font_medium.render("Call reset() to play again", True, self.COLOR_TEXT)
            reset_rect = reset_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 40))
            self.screen.blit(reset_surf, reset_rect)

    def _respawn_visuals(self):
        self.hunters = []
        for _ in range(self.num_hunters):
            self._add_hunter_visual()
        
        self.centipede_particles = []
        num_particles = min(self.centipede_population, 300) # Cap for performance
        for _ in range(num_particles):
            self._add_centipede_particle()

    def _add_hunter_visual(self):
        hunter = {
            'pos': [self.np_random.uniform(20, self.WIDTH - 20), self.HEIGHT * 0.4 - 15],
            'target_x': self.np_random.uniform(20, self.WIDTH - 20),
            'speed': self.np_random.uniform(0.5, 1.5)
        }
        self.hunters.append(hunter)

    def _add_centipede_particle(self):
        particle = {
            'pos': [
                self.np_random.uniform(0, self.WIDTH),
                self.np_random.uniform(self.HEIGHT * 0.4, self.HEIGHT)
            ]
        }
        self.centipede_particles.append(particle)

    def _add_floating_text(self, text, pos, color):
        ft = {
            'text': text,
            'pos': list(pos),
            'color': color,
            'life': 60, # frames
            'alpha': 255
        }
        self.floating_texts.append(ft)

    def _update_animations(self):
        # Update hunters
        for h in self.hunters:
            h['pos'][0] += (h['target_x'] - h['pos'][0]) * 0.02 * h['speed']
            if abs(h['pos'][0] - h['target_x']) < 5:
                h['target_x'] = self.np_random.uniform(20, self.WIDTH - 20)
        
        # Update floating texts
        for ft in self.floating_texts[:]:
            ft['pos'][1] -= 0.5
            ft['life'] -= 1
            ft['alpha'] = max(0, int(255 * (ft['life'] / 60)))
            if ft['life'] <= 0:
                self.floating_texts.remove(ft)

    def render(self):
        if self.metadata["render_modes"][0] == "rgb_array":
            return self._get_observation()

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # It will not be run by the evaluation server.
    env = GameEnv()
    obs, info = env.reset()
    
    # We need to set the video driver for Pygame to work without a display.
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        # For manual testing, we want to see the game.
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Centipede Exterminator Tycoon - Manual Test")
    clock = pygame.time.Clock()
    
    running = True
    game_over_flag = False
    
    while running:
        # Default action is "wait"
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and game_over_flag:
                obs, info = env.reset()
                game_over_flag = False
                continue # Skip the rest of the loop for the reset frame

        if not game_over_flag:
            keys = pygame.key.get_pressed()
            
            # This is a continuous time game, so we process actions and step every frame.
            # The agent can choose to do nothing by passing [0,0,0]
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            if keys[pygame.K_SPACE]:
                action[1] = 1
                
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                game_over_flag = True
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata["render_fps"])

    env.close()